"""
认知负荷数据采集系统 - AOI (Area of Interest) 版本

功能：
1. 实时眼动追踪
2. 活动窗口检测（洛谷页面 vs 其他窗口）
3. AOI 区域判断（题目标题/描述/示例/约束/代码区）
4. 数据记录与导出

逻辑：
- 如果当前窗口是洛谷页面 → 判断视线在哪个 AOI 区域
- 如果当前窗口不是洛谷页面 → 标记为 "CODE_EDITOR"（代码区）

依赖：
pip install mediapipe opencv-python pyautogui pygetwindow numpy
"""

import cv2
import numpy as np
import time
import os
import json
import csv
import ctypes
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
import threading

# 导入依赖
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("请安装 mediapipe: pip install mediapipe")
    exit(1)

try:
    import pygetwindow as gw
except ImportError:
    print("请安装 pygetwindow: pip install pygetwindow")
    gw = None

try:
    import pyautogui
except ImportError:
    print("请安装 pyautogui: pip install pyautogui")
    exit(1)


# ==================== 配置 ====================

# 洛谷页面识别关键词
LUOGU_KEYWORDS = ["洛谷", "luogu", "P1000", "P1001", "题目", "problem"]

# 默认 AOI 区域配置（相对于洛谷页面的比例坐标）
# 可以通过 aoi_config_tool.py 重新配置
DEFAULT_AOI_CONFIG = {
    "A_TITLE": {
        "name": "题目标题",
        "x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.08,
        "color": (255, 100, 100)
    },
    "B_PROBLEM": {
        "name": "题目描述",
        "x1": 0.0, "y1": 0.08, "x2": 0.5, "y2": 0.35,
        "color": (100, 255, 100)
    },
    "C_IO_FORMAT": {
        "name": "输入输出格式",
        "x1": 0.0, "y1": 0.35, "x2": 0.5, "y2": 0.50,
        "color": (100, 100, 255)
    },
    "D_EXAMPLES": {
        "name": "示例",
        "x1": 0.0, "y1": 0.50, "x2": 0.5, "y2": 0.75,
        "color": (255, 255, 100)
    },
    "E_CONSTRAINTS": {
        "name": "约束/提示",
        "x1": 0.0, "y1": 0.75, "x2": 0.5, "y2": 1.0,
        "color": (255, 100, 255)
    },
    "F_CODE_EDITOR": {
        "name": "代码区",
        "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,  # 非洛谷窗口默认全屏为代码区
        "color": (100, 255, 255)
    }
}


# ==================== 数据结构 ====================

@dataclass
class GazeRecord:
    """单条视线记录"""
    timestamp: float
    gaze_x: float  # 屏幕归一化坐标 0-1
    gaze_y: float
    screen_x: int  # 屏幕像素坐标
    screen_y: int
    aoi_region: str  # AOI 区域标识
    aoi_name: str    # AOI 区域名称
    window_title: str  # 当前活动窗口
    is_luogu: bool   # 是否在洛谷页面
    pupil_diameter: float
    is_fixation: bool
    fixation_id: int
    task_id: str
    event_marker: str  # 事件标记


@dataclass
class FixationRecord:
    """注视事件记录"""
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
    """AOI 转换记录"""
    timestamp: float
    from_aoi: str
    to_aoi: str
    task_id: str


@dataclass
class SessionData:
    """会话数据"""
    session_id: str
    start_time: float
    gaze_records: List[GazeRecord] = field(default_factory=list)
    fixations: List[FixationRecord] = field(default_factory=list)
    transitions: List[AOITransition] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)


# ==================== 滤波器 ====================

class OneEuroFilter:
    """一欧元滤波器"""
    
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
    """活动窗口检测器"""
    
    def __init__(self, keywords: List[str] = None):
        self.keywords = keywords or LUOGU_KEYWORDS
        self._last_title = ""
        self._last_is_luogu = False
    
    def get_active_window_title(self) -> str:
        """获取当前活动窗口标题"""
        try:
            if gw:
                window = gw.getActiveWindow()
                if window:
                    return window.title
            # 备用方案：使用 ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value
        except:
            return ""
    
    def is_luogu_window(self, title: str = None) -> bool:
        """判断是否是洛谷页面"""
        if title is None:
            title = self.get_active_window_title()
        
        title_lower = title.lower()
        for keyword in self.keywords:
            if keyword.lower() in title_lower:
                return True
        return False
    
    def get_window_info(self) -> Tuple[str, bool]:
        """获取窗口信息"""
        title = self.get_active_window_title()
        is_luogu = self.is_luogu_window(title)
        
        # 缓存以减少检测频率
        self._last_title = title
        self._last_is_luogu = is_luogu
        
        return title, is_luogu


# ==================== AOI 管理器 ====================

class AOIManager:
    """AOI 区域管理器"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "aoi_config.json"
        self.aoi_config = self._load_config()
        self.current_aoi = None
        self.last_aoi = None
        self.aoi_enter_time = {}
    
    def _load_config(self) -> Dict:
        """加载 AOI 配置"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"已加载 AOI 配置: {self.config_path}")
                return config
            except:
                pass
        print("使用默认 AOI 配置")
        return DEFAULT_AOI_CONFIG.copy()
    
    def save_config(self):
        """保存 AOI 配置"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.aoi_config, f, ensure_ascii=False, indent=2)
        print(f"AOI 配置已保存: {self.config_path}")
    
    def get_aoi_at_position(self, x: float, y: float, is_luogu: bool) -> Tuple[str, str]:
        """
        根据视线位置和窗口状态判断 AOI 区域
        
        Args:
            x, y: 归一化屏幕坐标 (0-1)
            is_luogu: 是否在洛谷页面
        
        Returns:
            (aoi_id, aoi_name)
        """
        if not is_luogu:
            # 不在洛谷页面 → 代码区
            return "F_CODE_EDITOR", "代码区"
        
        # 在洛谷页面 → 判断具体 AOI
        for aoi_id, aoi_info in self.aoi_config.items():
            if aoi_id == "F_CODE_EDITOR":
                continue  # 跳过代码区（非洛谷时才用）
            
            x1, y1 = aoi_info.get("x1", 0), aoi_info.get("y1", 0)
            x2, y2 = aoi_info.get("x2", 1), aoi_info.get("y2", 1)
            
            if x1 <= x <= x2 and y1 <= y <= y2:
                return aoi_id, aoi_info.get("name", aoi_id)
        
        # 未匹配任何区域
        return "UNKNOWN", "未知区域"
    
    def update_aoi(self, aoi_id: str) -> Optional[Tuple[str, str]]:
        """
        更新当前 AOI，检测转换
        
        Returns:
            如果发生转换，返回 (from_aoi, to_aoi)，否则返回 None
        """
        transition = None
        
        if self.current_aoi != aoi_id:
            if self.current_aoi is not None:
                transition = (self.current_aoi, aoi_id)
            self.last_aoi = self.current_aoi
            self.current_aoi = aoi_id
            self.aoi_enter_time[aoi_id] = time.time()
        
        return transition


# ==================== 注视检测器 ====================

class FixationDetector:
    """注视检测器"""
    
    def __init__(self, velocity_threshold=0.02, duration_threshold=0.1):
        self.velocity_threshold = velocity_threshold  # 速度阈值
        self.duration_threshold = duration_threshold  # 最小注视时长
        
        self.is_fixating = False
        self.fixation_start = None
        self.fixation_points = []
        self.fixation_count = 0
        self.last_pos = None
        self.last_time = None
    
    def update(self, x: float, y: float, t: float) -> Optional[FixationRecord]:
        """
        更新注视状态
        
        Returns:
            如果注视结束，返回 FixationRecord，否则返回 None
        """
        completed_fixation = None
        
        if self.last_pos is not None and self.last_time is not None:
            dt = t - self.last_time
            if dt > 0:
                dx = x - self.last_pos[0]
                dy = y - self.last_pos[1]
                velocity = np.sqrt(dx**2 + dy**2) / dt
                
                if velocity < self.velocity_threshold:
                    # 低速 → 注视中
                    if not self.is_fixating:
                        self.is_fixating = True
                        self.fixation_start = t
                        self.fixation_points = []
                    self.fixation_points.append((x, y))
                else:
                    # 高速 → 扫视，注视结束
                    if self.is_fixating:
                        duration = t - self.fixation_start
                        if duration >= self.duration_threshold and self.fixation_points:
                            self.fixation_count += 1
                            points = np.array(self.fixation_points)
                            completed_fixation = FixationRecord(
                                fixation_id=self.fixation_count,
                                start_time=self.fixation_start,
                                end_time=t,
                                duration=duration,
                                center_x=np.mean(points[:, 0]),
                                center_y=np.mean(points[:, 1]),
                                aoi_region="",  # 由调用者填充
                                task_id=""
                            )
                        self.is_fixating = False
        
        self.last_pos = (x, y)
        self.last_time = t
        
        return completed_fixation


# ==================== 辅助函数 ====================

def download_model():
    """下载 MediaPipe 模型"""
    import urllib.request
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "face_landmarker.task")
    
    if os.path.exists(model_path):
        return model_path
    
    print("正在下载模型...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    try:
        urllib.request.urlretrieve(url, model_path)
        print("下载完成")
        return model_path
    except Exception as e:
        print(f"下载失败: {e}")
        return None


def get_iris_center(landmarks, indices):
    """获取虹膜中心"""
    points = []
    for idx in indices:
        if idx < len(landmarks):
            points.append([landmarks[idx].x, landmarks[idx].y])
    if not points:
        return None, None
    points = np.array(points)
    return np.mean(points[:, 0]), np.mean(points[:, 1])


def get_pupil_diameter(landmarks, indices):
    """估算瞳孔直径"""
    if len(indices) < 4:
        return 0.0
    try:
        points = [[landmarks[idx].x, landmarks[idx].y] for idx in indices if idx < len(landmarks)]
        if len(points) < 4:
            return 0.0
        points = np.array(points)
        width = np.max(points[:, 0]) - np.min(points[:, 0])
        height = np.max(points[:, 1]) - np.min(points[:, 1])
        return (width + height) / 2
    except:
        return 0.0


# 关键点索引
LEFT_IRIS = [473, 474, 475, 476, 477]
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]


# ==================== 数据采集器 ====================

class CognitiveLoadCollector:
    """认知负荷数据采集器"""
    
    def __init__(self, output_dir: str = "cognitive_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 会话信息
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_data = SessionData(
            session_id=self.session_id,
            start_time=time.time()
        )
        
        # 当前任务
        self.current_task_id = "task_0"
        self.task_start_time = time.time()
        
        # 组件
        self.window_detector = WindowDetector()
        self.aoi_manager = AOIManager()
        self.fixation_detector = FixationDetector()
        self.filter_x = OneEuroFilter(freq=30, mincutoff=0.8, beta=0.01)
        self.filter_y = OneEuroFilter(freq=30, mincutoff=0.8, beta=0.01)
        
        # 状态
        self.recording = True
        self.last_aoi = None
        
        print(f"数据采集器初始化完成，会话ID: {self.session_id}")
    
    def add_event(self, event_type: str, description: str = ""):
        """添加事件标记"""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "task_id": self.current_task_id,
            "description": description
        }
        self.session_data.events.append(event)
        print(f"[事件] {event_type}: {description}")
    
    def start_task(self, task_id: str, description: str = ""):
        """开始新任务"""
        self.current_task_id = task_id
        self.task_start_time = time.time()
        self.add_event("TASK_START", description)
    
    def end_task(self, result: str = ""):
        """结束当前任务"""
        self.add_event("TASK_END", result)
    
    def record_gaze(self, gaze_x: float, gaze_y: float, 
                    pupil_diameter: float, event_marker: str = ""):
        """记录一条视线数据"""
        if not self.recording:
            return
        
        t = time.time()
        
        # 滤波
        filtered_x = self.filter_x.update([gaze_x], t)[0]
        filtered_y = self.filter_y.update([gaze_y], t)[0]
        
        # 获取窗口信息
        window_title, is_luogu = self.window_detector.get_window_info()
        
        # 获取 AOI
        aoi_id, aoi_name = self.aoi_manager.get_aoi_at_position(
            filtered_x, filtered_y, is_luogu
        )
        
        # 检测 AOI 转换
        transition = self.aoi_manager.update_aoi(aoi_id)
        if transition:
            self.session_data.transitions.append(AOITransition(
                timestamp=t,
                from_aoi=transition[0],
                to_aoi=transition[1],
                task_id=self.current_task_id
            ))
        
        # 检测注视
        fixation = self.fixation_detector.update(filtered_x, filtered_y, t)
        is_fixation = self.fixation_detector.is_fixating
        fixation_id = self.fixation_detector.fixation_count
        
        if fixation:
            fixation.aoi_region = aoi_id
            fixation.task_id = self.current_task_id
            self.session_data.fixations.append(fixation)
        
        # 屏幕坐标
        screen_w, screen_h = pyautogui.size()
        screen_x = int(filtered_x * screen_w)
        screen_y = int(filtered_y * screen_h)
        
        # 创建记录
        record = GazeRecord(
            timestamp=t,
            gaze_x=filtered_x,
            gaze_y=filtered_y,
            screen_x=screen_x,
            screen_y=screen_y,
            aoi_region=aoi_id,
            aoi_name=aoi_name,
            window_title=window_title[:50],  # 截断过长标题
            is_luogu=is_luogu,
            pupil_diameter=pupil_diameter,
            is_fixation=is_fixation,
            fixation_id=fixation_id,
            task_id=self.current_task_id,
            event_marker=event_marker
        )
        
        self.session_data.gaze_records.append(record)
        
        return aoi_id, aoi_name, is_luogu
    
    def export_data(self):
        """导出所有数据"""
        session_dir = os.path.join(self.output_dir, self.session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # 1. 导出原始视线数据
        gaze_file = os.path.join(session_dir, "gaze_data.csv")
        with open(gaze_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'gaze_x', 'gaze_y', 'screen_x', 'screen_y',
                'aoi_region', 'aoi_name', 'window_title', 'is_luogu',
                'pupil_diameter', 'is_fixation', 'fixation_id', 
                'task_id', 'event_marker'
            ])
            for r in self.session_data.gaze_records:
                writer.writerow([
                    r.timestamp, r.gaze_x, r.gaze_y, r.screen_x, r.screen_y,
                    r.aoi_region, r.aoi_name, r.window_title, r.is_luogu,
                    r.pupil_diameter, r.is_fixation, r.fixation_id,
                    r.task_id, r.event_marker
                ])
        print(f"视线数据已导出: {gaze_file}")
        
        # 2. 导出注视数据
        fixation_file = os.path.join(session_dir, "fixations.csv")
        with open(fixation_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'fixation_id', 'start_time', 'end_time', 'duration',
                'center_x', 'center_y', 'aoi_region', 'task_id'
            ])
            for fix in self.session_data.fixations:
                writer.writerow([
                    fix.fixation_id, fix.start_time, fix.end_time, fix.duration,
                    fix.center_x, fix.center_y, fix.aoi_region, fix.task_id
                ])
        print(f"注视数据已导出: {fixation_file}")
        
        # 3. 导出 AOI 转换数据
        transition_file = os.path.join(session_dir, "aoi_transitions.csv")
        with open(transition_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'from_aoi', 'to_aoi', 'task_id'])
            for t in self.session_data.transitions:
                writer.writerow([t.timestamp, t.from_aoi, t.to_aoi, t.task_id])
        print(f"AOI转换数据已导出: {transition_file}")
        
        # 4. 导出事件数据
        event_file = os.path.join(session_dir, "events.csv")
        with open(event_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'type', 'task_id', 'description'])
            for e in self.session_data.events:
                writer.writerow([e['timestamp'], e['type'], e['task_id'], e['description']])
        print(f"事件数据已导出: {event_file}")
        
        # 5. 导出会话元数据
        meta_file = os.path.join(session_dir, "session_meta.json")
        meta = {
            "session_id": self.session_id,
            "start_time": self.session_data.start_time,
            "end_time": time.time(),
            "duration": time.time() - self.session_data.start_time,
            "total_gaze_records": len(self.session_data.gaze_records),
            "total_fixations": len(self.session_data.fixations),
            "total_transitions": len(self.session_data.transitions),
            "total_events": len(self.session_data.events)
        }
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"元数据已导出: {meta_file}")
        
        print(f"\n所有数据已导出到: {session_dir}")
        return session_dir


# ==================== 主程序 ====================

class AOIGazeTracker:
    """带 AOI 分析的视线追踪器"""
    
    def __init__(self):
        self.collector = CognitiveLoadCollector()
        self.running = False
        
        # 屏幕尺寸
        try:
            user32 = ctypes.windll.user32
            self.screen_w = user32.GetSystemMetrics(0)
            self.screen_h = user32.GetSystemMetrics(1)
        except:
            self.screen_w, self.screen_h = 1920, 1080
        
        print(f"屏幕尺寸: {self.screen_w}x{self.screen_h}")
    
    def run(self):
        """运行追踪器"""
        # 下载模型
        model_path = download_model()
        if not model_path:
            return
        
        # 初始化检测器
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        try:
            detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"检测器初始化失败: {e}")
            return
        
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "=" * 60)
        print("认知负荷数据采集系统 - AOI 版本")
        print("=" * 60)
        print("\n快捷键:")
        print("  Q      - 退出并导出数据")
        print("  1-5    - 开始任务 1-5")
        print("  E      - 结束当前任务")
        print("  SPACE  - 添加自定义事件标记")
        print("  R      - 暂停/继续记录")
        print("  C      - 校准")
        print("=" * 60)
        
        self.running = True
        self.collector.add_event("SESSION_START", "数据采集开始")
        
        # 状态变量
        calibration = None
        show_aoi = True
        frame_count = 0
        fps_time = time.time()
        fps = 0
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                result = detector.detect(mp_image)
                
                # FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    now = time.time()
                    fps = 10 / (now - fps_time)
                    fps_time = now
                
                current_aoi = "N/A"
                is_luogu = False
                
                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    
                    # 获取虹膜中心
                    left_x, left_y = get_iris_center(landmarks, LEFT_IRIS)
                    right_x, right_y = get_iris_center(landmarks, RIGHT_IRIS)
                    
                    if left_x is None:
                        left_x, left_y = get_iris_center(landmarks, LEFT_EYE)
                        right_x, right_y = get_iris_center(landmarks, RIGHT_EYE)
                    
                    if left_x is not None and right_x is not None:
                        gaze_x = (left_x + right_x) / 2
                        gaze_y = (left_y + right_y) / 2
                        
                        # 简单映射（可以添加校准）
                        mapped_x = 0.5 + (gaze_x - 0.5) * 2.5
                        mapped_y = 0.5 + (gaze_y - 0.5) * 2.5
                        mapped_x = max(0.0, min(1.0, mapped_x))
                        mapped_y = max(0.0, min(1.0, mapped_y))
                        
                        # 瞳孔直径
                        pupil = get_pupil_diameter(landmarks, LEFT_IRIS)
                        
                        # 记录数据
                        aoi_id, aoi_name, is_luogu = self.collector.record_gaze(
                            mapped_x, mapped_y, pupil
                        )
                        current_aoi = aoi_name
                        
                        # 绘制眼睛位置
                        eye_px = int(gaze_x * frame.shape[1])
                        eye_py = int(gaze_y * frame.shape[0])
                        cv2.circle(frame, (eye_px, eye_py), 5, (0, 255, 0), -1)
                
                # 绘制界面
                # 状态信息
                status_color = (0, 255, 0) if self.collector.recording else (0, 0, 255)
                status_text = "Recording" if self.collector.recording else "Paused"
                cv2.putText(frame, f"{status_text} | FPS: {fps:.1f}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                # 窗口状态
                window_status = "洛谷页面" if is_luogu else "代码区(其他窗口)"
                cv2.putText(frame, f"Window: {window_status}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # AOI 信息
                cv2.putText(frame, f"AOI: {current_aoi}", (10, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 任务信息
                cv2.putText(frame, f"Task: {self.collector.current_task_id}", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
                
                # 数据统计
                stats_y = frame.shape[0] - 60
                cv2.putText(frame, f"Records: {len(self.collector.session_data.gaze_records)}", 
                           (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"Fixations: {len(self.collector.session_data.fixations)}", 
                           (10, stats_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"Transitions: {len(self.collector.session_data.transitions)}", 
                           (10, stats_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                cv2.imshow("AOI Gaze Tracker", frame)
                
                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    self.collector.add_event("SESSION_END", "数据采集结束")
                    break
                elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                    task_num = chr(key)
                    self.collector.start_task(f"task_{task_num}", f"开始任务 {task_num}")
                elif key == ord('e'):
                    self.collector.end_task("任务结束")
                elif key == 32:  # Space
                    self.collector.add_event("MARKER", "用户标记")
                elif key == ord('r'):
                    self.collector.recording = not self.collector.recording
                    status = "继续" if self.collector.recording else "暂停"
                    print(f"记录状态: {status}")
        
        except KeyboardInterrupt:
            print("\n中断")
        
        finally:
            # 导出数据
            self.collector.export_data()
            
            cap.release()
            cv2.destroyAllWindows()
            detector.close()


# ==================== 入口 ====================

if __name__ == "__main__":
    tracker = AOIGazeTracker()
    tracker.run()

