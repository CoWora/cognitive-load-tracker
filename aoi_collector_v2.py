"""
认知负荷数据采集系统 V2 - L2CS-Net 高精度版

特点：
1. 使用 L2CS-Net 进行高精度视线估计
2. 实时显示当前视线的 AOI 区域
3. 自动从窗口标题提取题号
4. 简化的任务流程（无需手动输入）
5. 完整的标签数据采集
"""

import cv2
import numpy as np
import time
import os
import json
import csv
import ctypes
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
from pathlib import Path

# 导入 L2CS-Net
try:
    import torch
    from l2cs import Pipeline
    L2CS_AVAILABLE = True
    print(f"L2CS-Net OK | PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    L2CS_AVAILABLE = False
    print(f"L2CS-Net 未安装: {e}")

try:
    import pygetwindow as gw
except ImportError:
    gw = None

try:
    import pyautogui
except ImportError:
    print("请安装 pyautogui")
    exit(1)


# ==================== 配置 ====================

# 洛谷页面识别关键词
LUOGU_KEYWORDS = ["洛谷", "luogu", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", 
                  "题目", "problem", "CF", "AT", "SP", "UVA"]

# 题号提取正则
PROBLEM_ID_PATTERN = re.compile(r'(P\d{4}|CF\d+[A-Z]?|AT_\w+|SP\w+|UVA\d+)', re.IGNORECASE)

# 默认 AOI 配置（洛谷页面左半边）
DEFAULT_AOI_CONFIG = {
    "A_TITLE": {"name": "题目标题", "x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.10, "color": (255, 100, 100)},
    "B_PROBLEM": {"name": "题目描述", "x1": 0.0, "y1": 0.10, "x2": 0.5, "y2": 0.40, "color": (100, 255, 100)},
    "C_IO_FORMAT": {"name": "输入输出格式", "x1": 0.0, "y1": 0.40, "x2": 0.5, "y2": 0.55, "color": (100, 100, 255)},
    "D_EXAMPLES": {"name": "示例", "x1": 0.0, "y1": 0.55, "x2": 0.5, "y2": 0.80, "color": (255, 255, 100)},
    "E_CONSTRAINTS": {"name": "约束/提示", "x1": 0.0, "y1": 0.80, "x2": 0.5, "y2": 1.0, "color": (255, 100, 255)},
    "F_CODE_EDITOR": {"name": "代码区", "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0, "color": (100, 255, 255)},
}

# 难度定义
DIFFICULTIES = ["EASY", "MEDIUM", "HARD", "EXPERT"]

# 结果定义
RESULTS = {
    'a': ("AC", "通过", (0, 255, 0)),
    'w': ("WA", "答案错误", (0, 0, 255)),
    't': ("TLE", "超时", (0, 165, 255)),
    'r': ("RE", "运行错误", (128, 0, 128)),
    'g': ("GIVEUP", "放弃", (128, 128, 128)),
}


# ==================== 数据结构 ====================

@dataclass
class TaskInfo:
    task_id: str
    problem_id: str = ""
    difficulty: str = "MEDIUM"
    start_time: float = 0.0
    end_time: float = 0.0
    result: str = ""
    subjective_difficulty: int = 0
    subjective_effort: int = 0


@dataclass
class GazeRecord:
    timestamp: float
    gaze_x: float
    gaze_y: float
    screen_x: int
    screen_y: int
    yaw: float
    pitch: float
    aoi_region: str
    aoi_name: str
    window_title: str
    is_luogu: bool
    is_fixation: bool
    fixation_id: int
    task_id: str


@dataclass
class FixationRecord:
    fixation_id: int
    start_time: float
    end_time: float
    duration: float
    center_x: float
    center_y: float
    aoi_region: str
    task_id: str


@dataclass
class AOITransition:
    timestamp: float
    from_aoi: str
    to_aoi: str
    task_id: str


@dataclass
class SessionData:
    session_id: str
    start_time: float
    participant_id: str = ""
    gaze_records: List[GazeRecord] = field(default_factory=list)
    fixations: List[FixationRecord] = field(default_factory=list)
    transitions: List[AOITransition] = field(default_factory=list)
    tasks: List[TaskInfo] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)


# ==================== 滤波器 ====================

class OneEuroFilter:
    def __init__(self, freq=30, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
    
    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
    
    def update(self, x, t=None):
        if t is None:
            t = time.time()
        x = np.array(x, dtype=np.float64)
        
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x
        
        dt = t - self.t_prev
        if dt <= 0:
            dt = 1.0 / self.freq
        self.freq = 1.0 / dt
        
        dx = (x - self.x_prev) / dt
        edx = self._alpha(self.dcutoff) * dx + (1 - self._alpha(self.dcutoff)) * self.dx_prev
        speed = np.abs(edx)
        cutoff = self.mincutoff + self.beta * speed
        result = self._alpha(np.mean(cutoff)) * x + (1 - self._alpha(np.mean(cutoff))) * self.x_prev
        
        self.x_prev = result.copy()
        self.dx_prev = edx.copy()
        self.t_prev = t
        return result
    
    def reset(self):
        self.x_prev = None


# ==================== 窗口检测 ====================

class WindowDetector:
    def __init__(self, keywords=None):
        self.keywords = keywords or LUOGU_KEYWORDS
        self._cache_time = 0
        self._cache_title = ""
        self._cache_is_luogu = False
        self._cache_problem_id = ""
    
    def get_active_window_title(self) -> str:
        try:
            if gw:
                window = gw.getActiveWindow()
                if window:
                    return window.title
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value
        except:
            return ""
    
    def is_luogu_window(self, title: str) -> bool:
        title_lower = title.lower()
        for keyword in self.keywords:
            if keyword.lower() in title_lower:
                return True
        return False
    
    def extract_problem_id(self, title: str) -> str:
        """从窗口标题提取题号"""
        match = PROBLEM_ID_PATTERN.search(title)
        if match:
            return match.group(1).upper()
        return ""
    
    def get_window_info(self) -> Tuple[str, bool, str]:
        """返回 (标题, 是否洛谷, 题号)"""
        now = time.time()
        # 缓存 0.2 秒，减少检测频率
        if now - self._cache_time < 0.2:
            return self._cache_title, self._cache_is_luogu, self._cache_problem_id
        
        title = self.get_active_window_title()
        is_luogu = self.is_luogu_window(title)
        problem_id = self.extract_problem_id(title) if is_luogu else ""
        
        self._cache_time = now
        self._cache_title = title
        self._cache_is_luogu = is_luogu
        self._cache_problem_id = problem_id
        
        return title, is_luogu, problem_id


# ==================== AOI 管理器 ====================

class AOIManager:
    def __init__(self, config_path="aoi_config.json"):
        self.config_path = config_path
        self.aoi_config = self._load_config()
        self.current_aoi = None
    
    def _load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return DEFAULT_AOI_CONFIG.copy()
    
    def get_aoi_at_position(self, x: float, y: float, is_luogu: bool) -> Tuple[str, str, Tuple]:
        """返回 (aoi_id, aoi_name, color)"""
        if not is_luogu:
            cfg = self.aoi_config.get("F_CODE_EDITOR", {})
            return "F_CODE_EDITOR", "代码区", tuple(cfg.get("color", [100, 255, 255]))
        
        for aoi_id, info in self.aoi_config.items():
            if aoi_id == "F_CODE_EDITOR":
                continue
            x1, y1 = info.get("x1", 0), info.get("y1", 0)
            x2, y2 = info.get("x2", 1), info.get("y2", 1)
            if x1 <= x <= x2 and y1 <= y <= y2:
                color = info.get("color", [200, 200, 200])
                if isinstance(color, list):
                    color = tuple(color)
                return aoi_id, info.get("name", aoi_id), color
        
        return "UNKNOWN", "未知区域", (128, 128, 128)
    
    def update_aoi(self, aoi_id: str) -> Optional[Tuple[str, str]]:
        transition = None
        if self.current_aoi != aoi_id:
            if self.current_aoi is not None:
                transition = (self.current_aoi, aoi_id)
            self.current_aoi = aoi_id
        return transition


# ==================== 注视检测 ====================

class FixationDetector:
    def __init__(self, velocity_threshold=0.015, duration_threshold=0.1):
        self.velocity_threshold = velocity_threshold
        self.duration_threshold = duration_threshold
        self.is_fixating = False
        self.fixation_start = None
        self.fixation_points = []
        self.fixation_count = 0
        self.last_pos = None
        self.last_time = None
    
    def update(self, x: float, y: float, t: float) -> Optional[FixationRecord]:
        completed = None
        
        if self.last_pos is not None and self.last_time is not None:
            dt = t - self.last_time
            if dt > 0:
                dx = x - self.last_pos[0]
                dy = y - self.last_pos[1]
                velocity = np.sqrt(dx**2 + dy**2) / dt
                
                if velocity < self.velocity_threshold:
                    if not self.is_fixating:
                        self.is_fixating = True
                        self.fixation_start = t
                        self.fixation_points = []
                    self.fixation_points.append((x, y))
                else:
                    if self.is_fixating:
                        duration = t - self.fixation_start
                        if duration >= self.duration_threshold and self.fixation_points:
                            self.fixation_count += 1
                            pts = np.array(self.fixation_points)
                            completed = FixationRecord(
                                fixation_id=self.fixation_count,
                                start_time=self.fixation_start,
                                end_time=t,
                                duration=duration,
                                center_x=np.mean(pts[:, 0]),
                                center_y=np.mean(pts[:, 1]),
                                aoi_region="",
                                task_id=""
                            )
                        self.is_fixating = False
        
        self.last_pos = (x, y)
        self.last_time = t
        return completed


# ==================== 主观评价 ====================

def show_rating_dialog(screen_w, screen_h, task_id: str) -> Tuple[int, int]:
    """显示主观评价对话框，返回 (难度评分, 努力评分)"""
    cv2.namedWindow("Rating", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Rating", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    ratings = [3, 3]  # [difficulty, effort]
    current = 0
    
    while True:
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        canvas[:] = (25, 25, 35)
        
        # 标题
        cv2.putText(canvas, f"Task {task_id} Complete - Rate Your Experience", 
                   (screen_w//2 - 280, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        questions = [
            ("How DIFFICULT was this task?", "1=Very Easy  5=Very Hard"),
            ("How much EFFORT did you spend?", "1=Very Low  5=Very High")
        ]
        
        for qi, (question, hint) in enumerate(questions):
            y_base = 200 + qi * 200
            q_color = (0, 255, 255) if current == qi else (150, 150, 150)
            
            cv2.putText(canvas, question, (200, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.9, q_color, 2)
            cv2.putText(canvas, hint, (200, y_base + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
            
            # 评分按钮
            for i in range(1, 6):
                x = 200 + (i - 1) * 140
                y = y_base + 60
                is_selected = ratings[qi] == i
                color = (0, 200, 255) if is_selected else (60, 60, 70)
                cv2.rectangle(canvas, (x, y), (x + 120, y + 60), color, -1 if is_selected else 2)
                text_color = (0, 0, 0) if is_selected else (200, 200, 200)
                cv2.putText(canvas, str(i), (x + 50, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
        
        # 提示
        cv2.putText(canvas, "Press 1-5 to rate | Tab to switch | Enter to confirm", 
                   (screen_w//2 - 250, screen_h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        
        cv2.imshow("Rating", canvas)
        key = cv2.waitKey(30) & 0xFF
        
        if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            ratings[current] = int(chr(key))
        elif key == 9:  # Tab
            current = 1 - current
        elif key == 13:  # Enter
            break
        elif key == 27:  # Esc
            break
    
    cv2.destroyWindow("Rating")
    return ratings[0], ratings[1]


# ==================== 数据采集器 ====================

class CognitiveLoadCollector:
    def __init__(self, output_dir="cognitive_data", participant_id=""):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_data = SessionData(
            session_id=self.session_id,
            start_time=time.time(),
            participant_id=participant_id
        )
        
        self.current_task: Optional[TaskInfo] = None
        self.task_count = 0
        self.current_difficulty_idx = 1  # 默认 MEDIUM
        
        self.window_detector = WindowDetector()
        self.aoi_manager = AOIManager()
        self.fixation_detector = FixationDetector()
        self.filter_x = OneEuroFilter(freq=30, mincutoff=0.8, beta=0.01)
        self.filter_y = OneEuroFilter(freq=30, mincutoff=0.8, beta=0.01)
        
        self.recording = True
        self.last_detected_problem = ""
        
        print(f"会话ID: {self.session_id}")
    
    def add_event(self, event_type: str, desc: str = ""):
        self.session_data.events.append({
            "timestamp": time.time(),
            "type": event_type,
            "task_id": self.current_task.task_id if self.current_task else "none",
            "description": desc
        })
    
    def start_task(self, problem_id: str = ""):
        """开始新任务"""
        if self.current_task and self.current_task.end_time == 0:
            # 自动结束未完成的任务
            self.current_task.end_time = time.time()
            self.current_task.result = "INTERRUPTED"
            self.session_data.tasks.append(self.current_task)
        
        self.task_count += 1
        task_id = f"task_{self.task_count:03d}"
        difficulty = DIFFICULTIES[self.current_difficulty_idx]
        
        self.current_task = TaskInfo(
            task_id=task_id,
            problem_id=problem_id,
            difficulty=difficulty,
            start_time=time.time()
        )
        
        self.add_event("TASK_START", f"{problem_id} | {difficulty}")
        print(f"\n▶ 开始任务: {task_id} | 题目: {problem_id} | 难度: {difficulty}")
    
    def end_task(self, result: str, screen_w: int, screen_h: int):
        """结束任务"""
        if not self.current_task:
            return
        
        self.current_task.end_time = time.time()
        self.current_task.result = result
        
        # 主观评价
        diff_rating, effort_rating = show_rating_dialog(screen_w, screen_h, self.current_task.task_id)
        self.current_task.subjective_difficulty = diff_rating
        self.current_task.subjective_effort = effort_rating
        
        duration = self.current_task.end_time - self.current_task.start_time
        self.add_event("TASK_END", f"{result} | {duration:.1f}s | 主观:{diff_rating},{effort_rating}")
        
        print(f"■ 结束任务: {self.current_task.task_id} | 结果: {result} | 用时: {duration:.1f}s")
        
        self.session_data.tasks.append(self.current_task)
        self.current_task = None
    
    def record_gaze(self, gaze_x: float, gaze_y: float, yaw: float, pitch: float,
                    window_title: str, is_luogu: bool, screen_w: int, screen_h: int):
        """记录视线数据"""
        if not self.recording:
            return None, None, None
        
        t = time.time()
        
        # 滤波
        fx = self.filter_x.update([gaze_x], t)[0]
        fy = self.filter_y.update([gaze_y], t)[0]
        
        # AOI 判断
        aoi_id, aoi_name, aoi_color = self.aoi_manager.get_aoi_at_position(fx, fy, is_luogu)
        
        # AOI 转换
        transition = self.aoi_manager.update_aoi(aoi_id)
        if transition and self.current_task:
            self.session_data.transitions.append(AOITransition(
                timestamp=t,
                from_aoi=transition[0],
                to_aoi=transition[1],
                task_id=self.current_task.task_id
            ))
        
        # 注视检测
        fixation = self.fixation_detector.update(fx, fy, t)
        is_fixation = self.fixation_detector.is_fixating
        fixation_id = self.fixation_detector.fixation_count
        
        if fixation and self.current_task:
            fixation.aoi_region = aoi_id
            fixation.task_id = self.current_task.task_id
            self.session_data.fixations.append(fixation)
        
        # 屏幕坐标
        sx = int(fx * screen_w)
        sy = int(fy * screen_h)
        
        # 记录
        record = GazeRecord(
            timestamp=t,
            gaze_x=fx, gaze_y=fy,
            screen_x=sx, screen_y=sy,
            yaw=yaw, pitch=pitch,
            aoi_region=aoi_id,
            aoi_name=aoi_name,
            window_title=window_title[:50],
            is_luogu=is_luogu,
            is_fixation=is_fixation,
            fixation_id=fixation_id,
            task_id=self.current_task.task_id if self.current_task else "none"
        )
        self.session_data.gaze_records.append(record)
        
        return aoi_id, aoi_name, aoi_color
    
    def export_data(self):
        """导出数据"""
        session_dir = os.path.join(self.output_dir, self.session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # 视线数据
        with open(os.path.join(session_dir, "gaze_data.csv"), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'gaze_x', 'gaze_y', 'screen_x', 'screen_y', 'yaw', 'pitch',
                       'aoi_region', 'aoi_name', 'window_title', 'is_luogu', 'is_fixation', 'fixation_id', 'task_id'])
            for r in self.session_data.gaze_records:
                w.writerow([r.timestamp, r.gaze_x, r.gaze_y, r.screen_x, r.screen_y, r.yaw, r.pitch,
                           r.aoi_region, r.aoi_name, r.window_title, r.is_luogu, r.is_fixation, r.fixation_id, r.task_id])
        
        # 注视数据
        with open(os.path.join(session_dir, "fixations.csv"), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['fixation_id', 'start_time', 'end_time', 'duration', 'center_x', 'center_y', 'aoi_region', 'task_id'])
            for r in self.session_data.fixations:
                w.writerow([r.fixation_id, r.start_time, r.end_time, r.duration, r.center_x, r.center_y, r.aoi_region, r.task_id])
        
        # AOI 转换
        with open(os.path.join(session_dir, "aoi_transitions.csv"), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'from_aoi', 'to_aoi', 'task_id'])
            for r in self.session_data.transitions:
                w.writerow([r.timestamp, r.from_aoi, r.to_aoi, r.task_id])
        
        # 任务数据（标签）
        with open(os.path.join(session_dir, "tasks.csv"), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['task_id', 'problem_id', 'difficulty', 'start_time', 'end_time', 'duration',
                       'result', 'subjective_difficulty', 'subjective_effort'])
            for t in self.session_data.tasks:
                duration = t.end_time - t.start_time if t.end_time > 0 else 0
                w.writerow([t.task_id, t.problem_id, t.difficulty, t.start_time, t.end_time, duration,
                           t.result, t.subjective_difficulty, t.subjective_effort])
        
        # 事件
        with open(os.path.join(session_dir, "events.csv"), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'type', 'task_id', 'description'])
            for e in self.session_data.events:
                w.writerow([e['timestamp'], e['type'], e['task_id'], e['description']])
        
        # 元数据
        meta = {
            "session_id": self.session_id,
            "participant_id": self.session_data.participant_id,
            "start_time": self.session_data.start_time,
            "end_time": time.time(),
            "duration": time.time() - self.session_data.start_time,
            "total_gaze_records": len(self.session_data.gaze_records),
            "total_fixations": len(self.session_data.fixations),
            "total_transitions": len(self.session_data.transitions),
            "total_tasks": len(self.session_data.tasks),
            "model": "L2CS-Net"
        }
        with open(os.path.join(session_dir, "session_meta.json"), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        print(f"\n数据已导出: {session_dir}")
        return session_dir


# ==================== 主程序 ====================

def main():
    print("=" * 70)
    print("认知负荷数据采集系统 V2 (L2CS-Net)")
    print("=" * 70)
    
    if not L2CS_AVAILABLE:
        print("错误：L2CS-Net 未安装")
        return
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 加载模型
    model_path = Path("models/L2CSNet_gaze360.pkl")
    if not model_path.exists():
        print(f"模型不存在: {model_path}")
        return
    
    print("加载 L2CS-Net...")
    try:
        gaze_pipeline = Pipeline(weights=str(model_path), arch='ResNet50', device=device)
        print("模型加载成功！")
    except Exception as e:
        print(f"加载失败: {e}")
        return
    
    # 摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头打开失败")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 屏幕尺寸
    try:
        user32 = ctypes.windll.user32
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
    except:
        screen_w, screen_h = 1920, 1080
    
    print(f"屏幕: {screen_w}x{screen_h}")
    
    # 采集器
    collector = CognitiveLoadCollector()
    
    # 参数
    yaw_range, pitch_range = 45, 35
    sensitivity = 1.2
    
    print("\n" + "=" * 70)
    print("快捷键:")
    print("  N       - 开始新任务（自动从窗口标题提取题号）")
    print("  1/2/3/4 - 设置难度: 简单/中等/困难/极难")
    print("  A       - 结束任务: AC (通过)")
    print("  W       - 结束任务: WA (答案错误)")
    print("  T       - 结束任务: TLE (超时)")
    print("  G       - 结束任务: 放弃")
    print("  P       - 暂停/继续记录")
    print("  Q       - 退出并保存")
    print("=" * 70)
    
    collector.add_event("SESSION_START", "")
    
    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # L2CS 推理
            results = gaze_pipeline.step(frame)
            
            # FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_time)
                fps_time = time.time()
            
            # 窗口检测
            window_title, is_luogu, detected_problem = collector.window_detector.get_window_info()
            
            # 默认值
            aoi_id, aoi_name, aoi_color = "N/A", "N/A", (128, 128, 128)
            yaw_val, pitch_val = 0, 0
            gaze_x, gaze_y = 0.5, 0.5
            
            if results is not None and results.pitch is not None and len(results.pitch) > 0:
                pitch_val = float(results.pitch[0])
                yaw_val = float(results.yaw[0])
                
                # 映射
                gaze_x = 0.5 + (yaw_val / yaw_range) * 0.5 * sensitivity
                gaze_y = 0.5 + (pitch_val / pitch_range) * 0.5 * sensitivity
                gaze_x = max(0.0, min(1.0, gaze_x))
                gaze_y = max(0.0, min(1.0, gaze_y))
                
                # 记录
                result = collector.record_gaze(
                    gaze_x, gaze_y, yaw_val, pitch_val,
                    window_title, is_luogu, screen_w, screen_h
                )
                if result[0]:
                    aoi_id, aoi_name, aoi_color = result
                
                # 绘制人脸框
                if results.bboxes is not None and len(results.bboxes) > 0:
                    bbox = results.bboxes[0]
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # ========== 绘制界面 ==========
            
            # 顶部信息栏
            cv2.rectangle(frame, (0, 0), (w, 130), (30, 30, 40), -1)
            
            # 录制状态
            rec_color = (0, 0, 255) if collector.recording else (128, 128, 128)
            cv2.circle(frame, (20, 20), 8, rec_color, -1)
            cv2.putText(frame, "REC" if collector.recording else "PAUSE", (35, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_color, 1)
            
            # FPS
            cv2.putText(frame, f"FPS: {fps:.0f}", (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 当前窗口类型
            window_type = "洛谷页面" if is_luogu else "代码区(其他)"
            wt_color = (100, 255, 100) if is_luogu else (255, 200, 100)
            cv2.putText(frame, f"Window: {window_type}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, wt_color, 1)
            
            # 检测到的题号
            if detected_problem:
                cv2.putText(frame, f"Problem: {detected_problem}", (200, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # ★★★ 当前 AOI（重点显示）★★★
            cv2.putText(frame, f"AOI: {aoi_name}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, aoi_color, 2)
            
            # 视线坐标
            cv2.putText(frame, f"Gaze: ({gaze_x:.2f}, {gaze_y:.2f})", (10, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            cv2.putText(frame, f"Yaw: {yaw_val:.1f} Pitch: {pitch_val:.1f}", (150, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            
            # 当前任务
            if collector.current_task:
                elapsed = time.time() - collector.current_task.start_time
                task_text = f"Task: {collector.current_task.task_id} | {collector.current_task.problem_id} | {collector.current_task.difficulty} | {elapsed:.0f}s"
                cv2.putText(frame, task_text, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1)
            else:
                cv2.putText(frame, "No task - Press N to start", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            
            # 右侧统计
            stats_x = w - 120
            cv2.putText(frame, f"Records: {len(collector.session_data.gaze_records)}", (stats_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(frame, f"Fixations: {len(collector.session_data.fixations)}", (stats_x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(frame, f"Trans: {len(collector.session_data.transitions)}", (stats_x, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(frame, f"Tasks: {len(collector.session_data.tasks)}", (stats_x, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            
            # 难度指示
            diff_name = DIFFICULTIES[collector.current_difficulty_idx]
            cv2.putText(frame, f"Diff: {diff_name}", (stats_x, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 100), 1)
            
            cv2.imshow("Cognitive Load Collector V2", frame)
            
            # 按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                collector.add_event("SESSION_END", "")
                break
            elif key == ord('n'):
                # 开始新任务，自动提取题号
                problem_id = detected_problem if detected_problem else f"unknown_{collector.task_count + 1}"
                collector.start_task(problem_id)
            elif key == ord('1'):
                collector.current_difficulty_idx = 0
                print(f"难度设置: {DIFFICULTIES[0]}")
            elif key == ord('2'):
                collector.current_difficulty_idx = 1
                print(f"难度设置: {DIFFICULTIES[1]}")
            elif key == ord('3'):
                collector.current_difficulty_idx = 2
                print(f"难度设置: {DIFFICULTIES[2]}")
            elif key == ord('4'):
                collector.current_difficulty_idx = 3
                print(f"难度设置: {DIFFICULTIES[3]}")
            elif key == ord('a') and collector.current_task:
                collector.end_task("AC", screen_w, screen_h)
            elif key == ord('w') and collector.current_task:
                collector.end_task("WA", screen_w, screen_h)
            elif key == ord('t') and collector.current_task:
                collector.end_task("TLE", screen_w, screen_h)
            elif key == ord('g') and collector.current_task:
                collector.end_task("GIVEUP", screen_w, screen_h)
            elif key == ord('r') and collector.current_task:
                collector.end_task("RE", screen_w, screen_h)
            elif key == ord('p'):
                collector.recording = not collector.recording
                print(f"记录: {'继续' if collector.recording else '暂停'}")
            elif key == 32:  # Space
                collector.add_event("MARKER", "用户标记")
                print("已添加标记")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        collector.export_data()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
