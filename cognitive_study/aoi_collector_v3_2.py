"""
认知负荷数据采集系统 V3 - 浏览器插件联动版

特点：
1. L2CS-Net 高精度视线估计 + 校准
2. 通过 WebSocket 接收浏览器插件的 AOI 坐标
3. 准确的题号和 AOI 检测
4. 完整的标签数据采集

需要：
1. 安装 luogu_extension 浏览器插件
2. 插件会通过 WebSocket 发送页面信息
"""

import cv2
import numpy as np
import time
import os
import json
import csv
import ctypes
import asyncio
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# ==================== 日志/输出控制 ====================
# 默认只输出“基础信息”（启动/任务开始结束/保存完成等），避免刷屏。
# - 完全静默：         $env:CLT_QUIET = "1"
# - 详细调试输出：     $env:CLT_VERBOSE = "1"
CLT_QUIET = os.getenv("CLT_QUIET", "0") == "1"
CLT_VERBOSE = os.getenv("CLT_VERBOSE", "0") == "1"

# 若用户误把 CLT_QUIET 设为 1，会导致“控制台看不到任何输出”。这里给出一次性提示。
if CLT_QUIET and (os.getenv("CLT_QUIET_NOTICE", "0") != "1"):
    # 不使用 _info（因为会被静默），直接提示一次
    print("[CLT] CLT_QUIET=1，已开启静默模式（控制台不会输出基础信息）。如需输出，请在终端执行：Remove-Item Env:CLT_QUIET 或 $env:CLT_QUIET='0'")

def _info(msg: str):
    if not CLT_QUIET:
        print(msg)

def _debug(msg: str):
    if (not CLT_QUIET) and CLT_VERBOSE:
        print(msg)

def _warn(msg: str):
    # 警告/错误仍然输出，避免用户无感知失败
    print(msg)

def _draw_help_overlay(frame, x: int = 10, bottom_margin: int = 12):
    """
    在主界面左下角绘制常驻操作提示，避免用户做完评分后忘记怎么操作。
    """
    lines = [
        "Keys:",
        "N: Start task   1-4: Difficulty",
        "A/W/T/G: End task   C: Calibrate",
        "B: Blink(test)  P: Pause  Q: Quit",
    ]

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    line_h = 18
    pad_x, pad_y = 10, 8

    max_text_w = 0
    for s in lines:
        (tw, th), _ = cv2.getTextSize(s, font, scale, thickness)
        max_text_w = max(max_text_w, tw)

    box_w = min(w - 2 * x, max_text_w + pad_x * 2)
    box_h = len(lines) * line_h + pad_y * 2
    y2 = h - bottom_margin
    y1 = max(0, y2 - box_h)
    x1 = x
    x2 = min(w, x1 + box_w)

    # 背景半透明
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 90), 1)

    ty = y1 + pad_y + 14
    for i, s in enumerate(lines):
        color = (240, 240, 240) if i == 0 else (200, 200, 200)
        cv2.putText(frame, s, (x1 + pad_x, ty), font, scale, color, thickness, cv2.LINE_AA)
        ty += line_h

# HTTP 服务器
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
HTTP_AVAILABLE = True

# L2CS-Net
try:
    import torch
    from l2cs import Pipeline
    L2CS_AVAILABLE = True
except ImportError:
    L2CS_AVAILABLE = False

try:
    import pyautogui
except ImportError:
    _warn("请安装 pyautogui")
    exit(1)

try:
    import pygetwindow as gw
    GW_AVAILABLE = True
except ImportError:
    GW_AVAILABLE = False
    _warn("pygetwindow 未安装，无法检测活动窗口")

# 代码编辑器关键词
CODE_EDITOR_KEYWORDS = ['cursor', 'vscode', 'visual studio code', 'pycharm', 'intellij', 
                        'sublime', 'atom', 'notepad', 'vim', 'neovim', 'code -']

def get_active_window_title() -> str:
    """获取当前活动窗口标题"""
    if not GW_AVAILABLE:
        return ""
    try:
        window = gw.getActiveWindow()
        if window:
            return window.title or ""
    except:
        pass
    return ""

def is_code_editor_active() -> bool:
    """仅判断是否处于代码编辑器窗口（Cursor/VSCode 等）。"""
    title = get_active_window_title().lower()
    if not title:
        return False
    return any(keyword in title for keyword in CODE_EDITOR_KEYWORDS)


# ==================== 数据结构 ====================

@dataclass
class AOIRegion:
    """AOI 区域信息（来自浏览器插件）"""
    aoi_id: str
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    found: bool = True


@dataclass
class PageInfo:
    """页面信息（来自浏览器插件）"""
    is_problem_page: bool = False
    problem_id: str = ""
    problem_title: str = ""
    difficulty: str = ""
    aoi_regions: Dict[str, AOIRegion] = field(default_factory=dict)
    last_update: float = 0


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
class BlinkRecord:
    timestamp: float
    blink_id: int
    ear: float
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
    gaze_records: List[GazeRecord] = field(default_factory=list)
    fixations: List[FixationRecord] = field(default_factory=list)
    transitions: List[AOITransition] = field(default_factory=list)
    blinks: List[BlinkRecord] = field(default_factory=list)
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


# ==================== 多项式校准 ====================

class GazeCalibration:
    """视线校准"""
    
    def __init__(self):
        self.calibrated = False
        self.coeffs_x = None
        self.coeffs_y = None
    
    def _features(self, yaw, pitch):
        """生成多项式特征"""
        return np.array([
            1, yaw, pitch, yaw**2, pitch**2, yaw*pitch
        ])
    
    def calibrate(self, gaze_points, screen_points):
        """
        校准
        gaze_points: [(yaw, pitch), ...]
        screen_points: [(screen_x, screen_y), ...]
        """
        if len(gaze_points) < 5:
            return False
        
        gaze = np.array(gaze_points)
        screen = np.array(screen_points)
        
        X = np.array([self._features(g[0], g[1]) for g in gaze])
        
        try:
            self.coeffs_x = np.linalg.lstsq(X, screen[:, 0], rcond=None)[0]
            self.coeffs_y = np.linalg.lstsq(X, screen[:, 1], rcond=None)[0]
            self.calibrated = True
            _info("校准成功！")
            return True
        except:
            return False
    
    def transform(self, yaw, pitch, screen_w, screen_h):
        """将 yaw/pitch 转换为屏幕坐标"""
        if self.calibrated:
            f = self._features(yaw, pitch)
            x = np.dot(f, self.coeffs_x)
            y = np.dot(f, self.coeffs_y)
        else:
            # 未校准时使用简单线性映射
            x = screen_w / 2 + (yaw / 45) * (screen_w / 2) * 1.2
            y = screen_h / 2 + (pitch / 35) * (screen_h / 2) * 1.2
        
        x = max(0, min(screen_w - 1, int(x)))
        y = max(0, min(screen_h - 1, int(y)))
        return x, y


# ==================== 注视检测 ====================

class FixationDetector:
    def __init__(self, distance_threshold=8, duration_threshold=0.1, max_duration=2.0):
        # 超低阈值以适应极度平滑的滤波
        self.distance_threshold = distance_threshold  # 像素
        self.duration_threshold = duration_threshold  # 最小注视时长
        self.max_duration = max_duration  # 最大注视时长（超过后自动分割）
        self.is_fixating = False
        self.fixation_start = None
        self.fixation_points = []
        self.fixation_count = 0
        self.last_pos = None
        self.last_time = None
    
    def update(self, x: int, y: int, t: float) -> Optional[FixationRecord]:
        completed = None
        
        if self.last_pos is not None:
            dx = x - self.last_pos[0]
            dy = y - self.last_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < self.distance_threshold:
                if not self.is_fixating:
                    self.is_fixating = True
                    self.fixation_start = t
                    self.fixation_points = []
                self.fixation_points.append((x, y))
                
                # 检查是否超过最大注视时长
                duration = t - self.fixation_start
                if duration >= self.max_duration and self.fixation_points:
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
                    # 重置开始新的注视
                    self.fixation_start = t
                    self.fixation_points = [(x, y)]
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


# ==================== 眨眼检测 ====================

class BlinkDetector:
    """使用 MediaPipe Face Mesh 进行眨眼检测"""
    
    # 眼睛关键点索引 (MediaPipe Face Mesh)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    def __init__(self, ear_threshold=0.21, consec_frames=2):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.blink_count = 0
        self.frame_counter = 0
        self.is_blinking = False
        self.last_blink_time = 0
        self.ear_history = []
        
        # 初始化 MediaPipe Face Mesh
        self.face_mesh = None
        self.available = False
        self._init_mediapipe()
    
    def _init_mediapipe(self):
        """
        尝试初始化 MediaPipe Face Mesh。
        - 若环境缺少 mediapipe 或初始化失败：自动禁用（available=False）
        - 仍可通过手动触发方式测试眨眼指标写入（见 manual_trigger）
        """
        try:
            import mediapipe as mp  # type: ignore

            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.available = True
        except Exception:
            self.face_mesh = None
            self.available = False

    def manual_trigger(self, t: float) -> dict:
        """手动触发一次眨眼事件（用于无法启用自动眨眼检测时的指标测试）。"""
        self.blink_count += 1
        self.last_blink_time = t
        # ear=-1 表示“手动触发/不可用真实 EAR”
        return {"timestamp": t, "blink_id": self.blink_count, "ear": -1.0}
    
    def _calculate_ear(self, eye_landmarks):
        """计算眼睛纵横比 (Eye Aspect Ratio)"""
        # 垂直距离
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        # 水平距离
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        # EAR
        if h == 0:
            return 0.3
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def update(self, frame, t: float) -> Optional[dict]:
        """更新眨眼检测，返回眨眼事件或 None"""
        if not self.available:
            return None
        
        try:
            # 转换颜色
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            
            # 提取眼睛关键点
            left_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in self.LEFT_EYE])
            right_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in self.RIGHT_EYE])
            
            # 计算双眼 EAR
            left_ear = self._calculate_ear(left_eye)
            right_ear = self._calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            self.ear_history.append(ear)
            if len(self.ear_history) > 10:
                self.ear_history.pop(0)
            
            # 检测眨眼
            blink_event = None
            if ear < self.ear_threshold:
                self.frame_counter += 1
            else:
                if self.frame_counter >= self.consec_frames:
                    self.blink_count += 1
                    self.last_blink_time = t
                    blink_event = {
                        'timestamp': t,
                        'blink_id': self.blink_count,
                        'ear': ear
                    }
                self.frame_counter = 0
            
            self.is_blinking = self.frame_counter > 0
            return blink_event
            
        except Exception as e:
            return None
    
    def get_current_ear(self) -> float:
        """获取当前 EAR 值"""
        if self.ear_history:
            return self.ear_history[-1]
        return 0.3


# ==================== HTTP 服务器 ====================

class AOIHTTPHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器"""
    
    server_instance = None  # 类变量，指向 AOIHTTPServer 实例
    
    def log_message(self, format, *args):
        pass  # 禁用日志输出
    
    def do_OPTIONS(self):
        """处理 CORS 预检请求"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        """处理 POST 请求"""
        if self.path == '/aoi':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            try:
                data = json.loads(body.decode('utf-8'))
                if self.server_instance:
                    self.server_instance._update_page_info(data)
                
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except Exception as e:
                self.send_response(400)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


class AOIHTTPServer:
    """接收浏览器插件数据的 HTTP 服务器"""
    
    def __init__(self, host='127.0.0.1', port=8766):
        self.host = host
        self.port = port
        self.page_info = PageInfo()
        self.connected = False
        self.server_thread = None
        self.running = False
        self.httpd = None
    
    def _update_page_info(self, data):
        """更新页面信息"""
        self.page_info.is_problem_page = data.get('is_problem_page', False)
        self.page_info.problem_id = data.get('problem_id', '') or ''
        self.page_info.problem_title = data.get('problem_title', '') or ''
        self.page_info.difficulty = data.get('difficulty', '') or ''
        self.page_info.last_update = time.time()
        
        if not self.connected:
            self.connected = True
            _debug("[HTTP] Browser plugin connected!")
        
        # 更新 AOI 区域
        aoi_data = data.get('aoi_regions', {})
        found_count = 0
        for aoi_id, info in aoi_data.items():
            if info.get('found', False):
                found_count += 1
                self.page_info.aoi_regions[aoi_id] = AOIRegion(
                    aoi_id=aoi_id,
                    name=self._get_aoi_name(aoi_id),
                    x1=info.get('x1', 0),
                    y1=info.get('y1', 0),
                    x2=info.get('x2', 0),
                    y2=info.get('y2', 0),
                    found=True
                )
        
        # 首次收到 AOI 数据时打印
        if found_count > 0 and not hasattr(self, '_aoi_logged'):
            self._aoi_logged = True
            _debug(f"[AOI] Found {found_count} regions")
    
    def _get_aoi_name(self, aoi_id):
        names = {
            'A_TITLE': 'Title',
            'B_PROBLEM': 'Problem',
            'C_IO_FORMAT': 'IO Format',
            'D_EXAMPLES': 'Examples',
            'E_CONSTRAINTS': 'Hints',
            'F_CODE_EDITOR': 'Code Editor'
        }
        return names.get(aoi_id, aoi_id)
    
    def start(self):
        """启动服务器"""
        def run_server():
            self.running = True
            AOIHTTPHandler.server_instance = self
            self.httpd = HTTPServer((self.host, self.port), AOIHTTPHandler)
            _debug(f"[HTTP] 服务器启动在 http://{self.host}:{self.port}")
            _debug("[HTTP] 等待浏览器插件连接...")
            while self.running:
                self.httpd.handle_request()
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
    
    def stop(self):
        self.running = False
        # 发送一个请求来解除 handle_request 的阻塞
        try:
            import urllib.request
            urllib.request.urlopen(f'http://{self.host}:{self.port}/shutdown', timeout=0.5)
        except:
            pass

    def is_luogu_active(self) -> bool:
        """只区分两种界面：洛谷（插件数据新鲜且为题目页） vs Code Editor（其他情况）。"""
        # 数据是否过期（超过 1 秒认为无效）
        if time.time() - self.page_info.last_update > 1.0:
            return False
        return bool(self.page_info.is_problem_page)
    
    def get_aoi_at_position(self, screen_x, screen_y) -> Tuple[str, str]:
        """根据屏幕坐标判断 AOI"""
        # 只区分：Code Editor vs 洛谷
        if is_code_editor_active():
            return "F_CODE_EDITOR", "Code Editor"

        if not self.is_luogu_active():
            # 非洛谷题目页（或插件数据过期）统一按 Code Editor 处理
            return "F_CODE_EDITOR", "Code Editor"
        
        # 检查各 AOI 区域
        for aoi_id, region in self.page_info.aoi_regions.items():
            if region.found:
                # 去除 AOI 框“宽度”约束：仅按纵向区间(y1~y2)判断区域
                if region.y1 <= screen_y <= region.y2:
                    return aoi_id, region.name
        
        # 在洛谷页面但不在已知区域
        return "UNKNOWN", "Other (Luogu)"


# ==================== 校准界面 ====================

def run_calibration(gaze_pipeline, cap, screen_w, screen_h, device):
    """运行校准"""
    
    # 9点校准
    targets = [
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
    ]
    
    gaze_points = []
    screen_points = []
    
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    samples_per_point = 30
    
    for idx, (tx, ty) in enumerate(targets, 1):
        target_x = int(tx * screen_w)
        target_y = int(ty * screen_h)
        samples = []
        
        while len(samples) < samples_per_point:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            results = gaze_pipeline.step(frame)
            
            yaw, pitch = None, None
            if results is not None and results.pitch is not None and len(results.pitch) > 0:
                yaw = float(results.yaw[0])
                pitch = float(results.pitch[0])
            
            # 绘制
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            canvas[:] = (20, 20, 30)
            
            # 目标点
            pulse = int(8 * np.sin(time.time() * 5))
            cv2.circle(canvas, (target_x, target_y), 35 + pulse, (0, 80, 150), 3)
            cv2.circle(canvas, (target_x, target_y), 20, (0, 0, 200), -1)
            cv2.circle(canvas, (target_x, target_y), 5, (255, 255, 255), -1)
            
            # 进度
            progress = len(samples) / samples_per_point
            bar_w = 300
            cv2.rectangle(canvas, ((screen_w-bar_w)//2, 60), ((screen_w+bar_w)//2, 68), (40, 40, 50), -1)
            cv2.rectangle(canvas, ((screen_w-bar_w)//2, 60), ((screen_w-bar_w)//2 + int(bar_w*progress), 68), (0, 200, 100), -1)
            
            cv2.putText(canvas, f"Point {idx}/9 - Look at the RED dot", (screen_w//2 - 180, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # 摄像头预览
            h, w = frame.shape[:2]
            pw = 160
            ph = int(h * pw / w)
            preview = cv2.resize(frame, (pw, ph))
            canvas[screen_h-ph-20:screen_h-20, (screen_w-pw)//2:(screen_w+pw)//2] = preview
            
            status = "Collecting..." if yaw is not None else "No face"
            color = (0, 200, 100) if yaw is not None else (0, 100, 200)
            cv2.putText(canvas, status, (screen_w//2 - 60, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            cv2.imshow("Calibration", canvas)
            
            if yaw is not None:
                samples.append((yaw, pitch))
            
            if cv2.waitKey(30) & 0xFF == 27:
                cv2.destroyWindow("Calibration")
                return None
        
        if samples:
            avg = np.mean(samples, axis=0)
            gaze_points.append(avg)
            screen_points.append((target_x, target_y))
    
    cv2.destroyWindow("Calibration")
    
    calibration = GazeCalibration()
    if calibration.calibrate(gaze_points, screen_points):
        return calibration
    return None


# ==================== 主观评价 ====================

def show_rating_dialog(screen_w, screen_h, task_id: str) -> Tuple[int, int]:
    cv2.namedWindow("Rating", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Rating", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    ratings = [3, 3]
    current = 0
    
    while True:
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        canvas[:] = (25, 25, 35)
        
        cv2.putText(canvas, f"Task Complete - Rate Your Experience", 
                   (screen_w//2 - 250, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        questions = [
            ("How DIFFICULT was this task?", "1=Very Easy  5=Very Hard"),
            ("How much EFFORT did you invest?", "1=Very Low  5=Very High")
        ]
        
        for qi, (question, hint) in enumerate(questions):
            y_base = 180 + qi * 180
            q_color = (0, 255, 255) if current == qi else (150, 150, 150)
            
            cv2.putText(canvas, question, (200, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.8, q_color, 2)
            cv2.putText(canvas, hint, (200, y_base + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
            
            for i in range(1, 6):
                x = 200 + (i - 1) * 130
                y = y_base + 50
                is_sel = ratings[qi] == i
                color = (0, 200, 255) if is_sel else (50, 50, 60)
                cv2.rectangle(canvas, (x, y), (x + 110, y + 55), color, -1 if is_sel else 2)
                tc = (0, 0, 0) if is_sel else (180, 180, 180)
                cv2.putText(canvas, str(i), (x + 45, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 1, tc, 2)
        
        cv2.putText(canvas, "1-5: Rate | Tab: Switch | Enter: Confirm", 
                   (screen_w//2 - 200, screen_h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        cv2.imshow("Rating", canvas)
        key = cv2.waitKey(30) & 0xFF
        
        if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            ratings[current] = int(chr(key))
        elif key == 9:
            current = 1 - current
        elif key == 13 or key == 27:
            break
    
    cv2.destroyWindow("Rating")
    return ratings[0], ratings[1]


# ==================== 数据采集器 ====================

class CognitiveLoadCollector:
    def __init__(self, output_dir="data/cognitive_study"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_data = SessionData(session_id=self.session_id, start_time=time.time())
        
        self.current_task: Optional[TaskInfo] = None
        self.task_count = 0
        self.current_difficulty_idx = 1
        
        self.fixation_detector = FixationDetector()
        self.blink_detector = BlinkDetector()  # 眨眼检测器
        self.filter_x = OneEuroFilter(freq=30, mincutoff=0.8, beta=0.01)
        self.filter_y = OneEuroFilter(freq=30, mincutoff=0.8, beta=0.01)
        
        self.current_aoi = None
        self.recording = True
    
    def process_blink(self, frame, t: float):
        """处理眨眼检测"""
        if not self.recording:
            return
        
        blink_event = self.blink_detector.update(frame, t)
        if blink_event:
            task_id = self.current_task.task_id if self.current_task else "none"
            self.session_data.blinks.append(BlinkRecord(
                timestamp=blink_event['timestamp'],
                blink_id=blink_event['blink_id'],
                ear=blink_event['ear'],
                task_id=task_id
            ))

    def add_manual_blink(self, t: Optional[float] = None):
        """手动添加一次眨眼事件（用于测试 blinks.csv 是否能正确写入）。"""
        if not self.recording:
            return
        if t is None:
            t = time.time()
        blink_event = self.blink_detector.manual_trigger(t)
        task_id = self.current_task.task_id if self.current_task else "none"
        self.session_data.blinks.append(BlinkRecord(
            timestamp=blink_event["timestamp"],
            blink_id=blink_event["blink_id"],
            ear=blink_event["ear"],
            task_id=task_id,
        ))
    
    def add_event(self, event_type: str, desc: str = "", task_id: Optional[str] = None):
        """记录事件；task_id 不传则自动取当前任务，否则可强制指定。"""
        self.session_data.events.append({
            "timestamp": time.time(),
            "type": event_type,
            "task_id": task_id if task_id is not None else (self.current_task.task_id if self.current_task else "none"),
            "description": desc
        })
    
    def start_task(self, problem_id: str, difficulty: str):
        if self.current_task and self.current_task.end_time == 0:
            self.current_task.end_time = time.time()
            self.current_task.result = "INTERRUPTED"
            self.session_data.tasks.append(self.current_task)
        
        self.task_count += 1
        self.current_task = TaskInfo(
            task_id=f"task_{self.task_count:03d}",
            problem_id=problem_id,
            difficulty=difficulty,
            start_time=time.time()
        )
        self.add_event("TASK_START", f"{self.current_task.task_id} | {problem_id} | {difficulty}", task_id=self.current_task.task_id)
        _info(f"▶ 任务开始: {self.current_task.task_id} | {problem_id} | {difficulty}")
    
    def end_task(self, result: str, screen_w: int, screen_h: int):
        if not self.current_task:
            return
        
        self.current_task.end_time = time.time()
        self.current_task.result = result
        
        d, e = show_rating_dialog(screen_w, screen_h, self.current_task.task_id)
        self.current_task.subjective_difficulty = d
        self.current_task.subjective_effort = e
        
        duration = self.current_task.end_time - self.current_task.start_time
        self.add_event("TASK_END", f"{self.current_task.task_id} | {result} | {duration:.1f}s | rating:{d},{e}", task_id=self.current_task.task_id)
        _info(f"■ 任务结束: {self.current_task.task_id} | {self.current_task.problem_id} | {self.current_task.difficulty} | {result} | {duration:.1f}s | 评分: {d},{e}")
        
        self.session_data.tasks.append(self.current_task)
        self.current_task = None

    def finalize_current_task(self, result: str = "QUIT"):
        """在退出/导出前兜底：若有正在进行的任务，补全 end_time 并写入 tasks/events。"""
        if not self.current_task:
            return
        if self.current_task.end_time and self.current_task.end_time > 0:
            return
        self.current_task.end_time = time.time()
        self.current_task.result = result
        duration = self.current_task.end_time - self.current_task.start_time
        # 不弹出评分窗口，避免退出流程卡住
        self.add_event("TASK_END", f"{self.current_task.task_id} | {result} | {duration:.1f}s | rating:NA", task_id=self.current_task.task_id)
        self.session_data.tasks.append(self.current_task)
        self.current_task = None
    
    def record_gaze(self, screen_x: int, screen_y: int, yaw: float, pitch: float,
                    aoi_id: str, aoi_name: str, is_luogu: bool):
        if not self.recording:
            return
        
        t = time.time()
        task_id = self.current_task.task_id if self.current_task else "none"
        
        # AOI 转换检测
        if self.current_aoi != aoi_id:
            if self.current_aoi is not None:
                self.session_data.transitions.append(AOITransition(
                    timestamp=t,
                    from_aoi=self.current_aoi,
                    to_aoi=aoi_id,
                    task_id=task_id
                ))
            self.current_aoi = aoi_id
        
        # 注视检测
        fixation = self.fixation_detector.update(screen_x, screen_y, t)
        is_fixation = self.fixation_detector.is_fixating
        fixation_id = self.fixation_detector.fixation_count
        
        if fixation:
            fixation.aoi_region = aoi_id
            fixation.task_id = task_id
            self.session_data.fixations.append(fixation)
        
        # 记录
        record = GazeRecord(
            timestamp=t,
            gaze_x=screen_x / pyautogui.size()[0],
            gaze_y=screen_y / pyautogui.size()[1],
            screen_x=screen_x,
            screen_y=screen_y,
            yaw=yaw,
            pitch=pitch,
            aoi_region=aoi_id,
            aoi_name=aoi_name,
            is_luogu=is_luogu,
            is_fixation=is_fixation,
            fixation_id=fixation_id,
            task_id=task_id
        )
        self.session_data.gaze_records.append(record)
    
    def export_data(self):
        # 兜底：如果用户未正常结束任务就退出，也要把任务写入 tasks.csv
        self.finalize_current_task(result="QUIT")
        session_dir = os.path.join(self.output_dir, self.session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # 视线数据
        with open(os.path.join(session_dir, "gaze_data.csv"), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'gaze_x', 'gaze_y', 'screen_x', 'screen_y', 'yaw', 'pitch',
                       'aoi_region', 'aoi_name', 'is_luogu', 'is_fixation', 'fixation_id', 'task_id'])
            for r in self.session_data.gaze_records:
                w.writerow([r.timestamp, r.gaze_x, r.gaze_y, r.screen_x, r.screen_y, r.yaw, r.pitch,
                           r.aoi_region, r.aoi_name, r.is_luogu, r.is_fixation, r.fixation_id, r.task_id])
        
        # 其他文件类似...
        with open(os.path.join(session_dir, "fixations.csv"), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['fixation_id', 'start_time', 'end_time', 'duration', 'center_x', 'center_y', 'aoi_region', 'task_id'])
            for r in self.session_data.fixations:
                w.writerow([r.fixation_id, r.start_time, r.end_time, r.duration, r.center_x, r.center_y, r.aoi_region, r.task_id])
        
        with open(os.path.join(session_dir, "aoi_transitions.csv"), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'from_aoi', 'to_aoi', 'task_id'])
            for r in self.session_data.transitions:
                w.writerow([r.timestamp, r.from_aoi, r.to_aoi, r.task_id])
        
        with open(os.path.join(session_dir, "tasks.csv"), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['task_id', 'problem_id', 'difficulty', 'start_time', 'end_time', 'duration',
                       'result', 'subjective_difficulty', 'subjective_effort'])
            for t in self.session_data.tasks:
                duration = t.end_time - t.start_time if t.end_time > 0 else 0
                w.writerow([t.task_id, t.problem_id, t.difficulty, t.start_time, t.end_time, duration,
                           t.result, t.subjective_difficulty, t.subjective_effort])
        
        with open(os.path.join(session_dir, "events.csv"), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'type', 'task_id', 'description'])
            for e in self.session_data.events:
                w.writerow([e['timestamp'], e['type'], e['task_id'], e['description']])
        
        # 眨眼数据
        with open(os.path.join(session_dir, "blinks.csv"), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['timestamp', 'blink_id', 'ear', 'task_id'])
            for b in self.session_data.blinks:
                w.writerow([b.timestamp, b.blink_id, b.ear, b.task_id])
        
        meta = {
            "session_id": self.session_id,
            "start_time": self.session_data.start_time,
            "end_time": time.time(),
            "total_gaze_records": len(self.session_data.gaze_records),
            "total_fixations": len(self.session_data.fixations),
            "total_transitions": len(self.session_data.transitions),
            "total_blinks": len(self.session_data.blinks),
            "total_tasks": len(self.session_data.tasks)
        }
        with open(os.path.join(session_dir, "session_meta.json"), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        
        _info(f"\n数据已导出: {session_dir}")


# ==================== 主程序 ====================

DIFFICULTIES = ["EASY", "MEDIUM", "HARD", "EXPERT"]

def main():
    _info("=" * 70)
    _info("认知负荷数据采集系统 V3.2 (眨眼可测试版)")
    _info("=" * 70)
    
    if not L2CS_AVAILABLE:
        _warn("错误: L2CS-Net 未安装")
        return
    
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _info(f"设备: {device}")
    
    # 屏幕
    try:
        user32 = ctypes.windll.user32
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
    except:
        screen_w, screen_h = 1920, 1080
    _info(f"屏幕: {screen_w}x{screen_h}")
    
    # 加载模型（同时兼容从不同工作目录启动）
    weights_name = "L2CSNet_gaze360.pkl"
    script_dir = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / "models" / weights_name,
        script_dir / "models" / weights_name,
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        _warn("模型不存在，已尝试以下路径：")
        for p in candidates:
            _warn(f"  - {p}")
        return
    
    _info("加载 L2CS-Net...")
    gaze_pipeline = Pipeline(weights=str(model_path), arch='ResNet50', device=device)
    _info("模型加载成功！")
    
    # 摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        _warn("摄像头打开失败")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 启动 HTTP 服务器（接收浏览器插件数据）
    aoi_server = AOIHTTPServer()
    aoi_server.start()
    
    # 采集器
    collector = CognitiveLoadCollector()
    
    # 校准
    _info("按 C 进行校准（推荐），或按任意键跳过...")
    calibration = None
    
    _debug("\n" + "=" * 70)
    _debug("Hotkeys:")
    _debug("  C       - Calibrate gaze")
    _debug("  +/-     - Adjust smoothing (+ = more smooth, - = less smooth)")
    _debug("  N       - Start new task")
    _debug("  1/2/3/4 - Set difficulty")
    _debug("  A/W/T/G - End task (AC/WA/TLE/GiveUp)")
    _debug("  B       - Manual blink (test blinks.csv)")
    _debug("  P       - Pause/Resume recording")
    _debug("  Q       - Quit and save")
    _debug("=" * 70)
    _debug("请确保已安装浏览器插件并打开洛谷页面！")
    
    collector.add_event("SESSION_START", "", task_id="none")
    
    # 视线平滑滤波器 (降低灵敏度和抖动)
    # mincutoff: 越小越平滑 (0.1-2.0)
    # beta: 对快速移动的响应 (0.001-0.1)
    filter_x = OneEuroFilter(freq=30, mincutoff=0.25, beta=0.001)  # 横向非常平滑
    filter_y = OneEuroFilter(freq=30, mincutoff=0.4, beta=0.005)   # 纵向响应稍快
    
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
            
            # L2CS
            # 注意：l2cs 的 Pipeline.step() 在“未检测到人脸”的帧上可能会对空列表 np.stack，
            # 进而抛 ValueError（need at least one array to stack）。这里做兼容处理，避免程序直接崩溃。
            try:
                results = gaze_pipeline.step(frame)
            except ValueError:
                results = None
            
            # FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_time)
                fps_time = time.time()
            
            # 从插件获取信息（仅用于“是否洛谷题目页”与题号）
            page_info = aoi_server.page_info
            is_luogu = aoi_server.is_luogu_active()
            problem_id = page_info.problem_id
            
            aoi_id, aoi_name = "N/A", "N/A"
            screen_x, screen_y = screen_w // 2, screen_h // 2
            yaw_val, pitch_val = 0, 0
            
            if results is not None and results.pitch is not None and len(results.pitch) > 0:
                yaw_val = float(results.yaw[0])
                pitch_val = float(results.pitch[0])
                
                # 转换为屏幕坐标
                # 横向灵敏度低，纵向灵敏度稍高（更容易检测上下移动）
                if calibration:
                    raw_x, raw_y = calibration.transform(yaw_val, pitch_val, screen_w, screen_h)
                else:
                    raw_x = int(screen_w / 2 + (yaw_val / 55) * (screen_w / 2) * 0.4)  # 横向很低
                    raw_y = int(screen_h / 2 + (pitch_val / 30) * (screen_h / 2) * 0.7)  # 纵向稍高
                
                # 应用平滑滤波 (大幅降低抖动)
                screen_x = int(filter_x.update(raw_x))
                screen_y = int(filter_y.update(raw_y))
                screen_x = max(0, min(screen_w - 1, screen_x))
                screen_y = max(0, min(screen_h - 1, screen_y))
                
                # 获取 AOI（使用插件数据）
                aoi_id, aoi_name = aoi_server.get_aoi_at_position(screen_x, screen_y)
                
                # 记录
                collector.record_gaze(screen_x, screen_y, yaw_val, pitch_val, aoi_id, aoi_name, is_luogu)
            
            # 眨眼检测（每帧都检测）
            collector.process_blink(frame, time.time())
            
            if results is not None and results.pitch is not None and len(results.pitch) > 0:
                # 绘制人脸框
                if results.bboxes is not None and len(results.bboxes) > 0:
                    bbox = results.bboxes[0]
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # ========== 绘制界面 ==========
            cv2.rectangle(frame, (0, 0), (w, 140), (30, 30, 40), -1)
            
            # 连接状态
            ws_status = "[Plugin OK]" if aoi_server.connected else "[No Plugin]"
            ws_color = (0, 255, 0) if aoi_server.connected else (100, 100, 100)
            cv2.putText(frame, ws_status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ws_color, 1)
            
            # 校准状态 - 未校准时强烈警告
            if calibration:
                cv2.putText(frame, "[Calibrated]", (120, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # 闪烁效果
                if int(time.time() * 2) % 2 == 0:
                    cv2.putText(frame, "!! PRESS C TO CALIBRATE !!", (120, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # FPS 和平滑度
            cv2.putText(frame, f"FPS:{fps:.0f}", (320, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"Smooth:{filter_x.mincutoff:.1f}", (380, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # 题号（从插件获取）
            if problem_id:
                cv2.putText(frame, f"Problem: {problem_id}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Problem: (waiting...)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # AOI（重点显示）
            aoi_colors = {
                'A_TITLE': (255, 100, 100),
                'B_PROBLEM': (100, 255, 100),
                'C_IO_FORMAT': (100, 100, 255),
                'D_EXAMPLES': (255, 255, 100),
                'E_CONSTRAINTS': (255, 100, 255),
                'F_CODE_EDITOR': (100, 255, 255),
            }
            aoi_color = aoi_colors.get(aoi_id, (200, 200, 200))
            cv2.putText(frame, f"AOI: {aoi_name}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, aoi_color, 2)
            
            # 视线坐标
            cv2.putText(frame, f"Screen: ({screen_x}, {screen_y})", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            cv2.putText(frame, f"Yaw: {yaw_val:.1f} Pitch: {pitch_val:.1f}", (180, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            
            # 当前任务
            if collector.current_task:
                elapsed = time.time() - collector.current_task.start_time
                cv2.putText(frame, f"Task: {collector.current_task.problem_id} | {elapsed:.0f}s", 
                           (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
            else:
                cv2.putText(frame, "No task (Press N)", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # 右侧统计
            cv2.putText(frame, f"Records: {len(collector.session_data.gaze_records)}", (w-150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            cv2.putText(frame, f"Fixations: {len(collector.session_data.fixations)}", (w-150, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
            # 眨眼次数和 EAR
            blink_count = len(collector.session_data.blinks)
            ear = collector.blink_detector.get_current_ear() if collector.blink_detector.available else 0
            blink_color = (100, 100, 255) if collector.blink_detector.is_blinking else (150, 150, 150)
            cv2.putText(frame, f"Blinks: {blink_count} EAR:{ear:.2f}", (w-150, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.35, blink_color, 1)
            # 显示当前界面类型：洛谷 vs Code Editor
            win_is_code = is_code_editor_active()
            win_is_luogu = aoi_server.is_luogu_active() and (not win_is_code)
            win_type = "Luogu" if win_is_luogu else "Code"
            win_color = (100, 255, 100) if win_is_luogu else (255, 255, 100)
            cv2.putText(frame, f"Win: {win_type}", (w-150, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.35, win_color, 1)
            
            # 显示 AOI 区域边界 (缩放到摄像头画面)
            scale_x = w / screen_w
            scale_y = h / screen_h
            aoi_colors_draw = {
                'A_TITLE': (255, 100, 100), 'B_PROBLEM': (100, 255, 100),
                'C_IO_FORMAT': (100, 100, 255), 'D_EXAMPLES': (255, 255, 100),
                'E_CONSTRAINTS': (255, 100, 255)
            }
            for region_id, region in aoi_server.page_info.aoi_regions.items():
                if region.found:
                    rx1 = int(region.x1 * scale_x)
                    ry1 = int(region.y1 * scale_y)
                    rx2 = int(region.x2 * scale_x)
                    ry2 = int(region.y2 * scale_y)
                    color = aoi_colors_draw.get(region_id, (128, 128, 128))
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 1)
                    cv2.putText(frame, region_id[:1], (rx1+2, ry1+12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # 显示视线位置 (缩放到摄像头画面)
            gaze_draw_x = int(screen_x * scale_x)
            gaze_draw_y = int(screen_y * scale_y)
            cv2.circle(frame, (gaze_draw_x, gaze_draw_y), 8, (0, 0, 255), -1)
            cv2.circle(frame, (gaze_draw_x, gaze_draw_y), 10, (255, 255, 255), 2)

            # 左下角常驻操作提示（防止评分后忘记怎么按）
            _draw_help_overlay(frame, x=10, bottom_margin=10)
            
            cv2.imshow("Cognitive Load Collector V3.2", frame)
            
            # 按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                collector.add_event("SESSION_END", "", task_id="none")
                break
            elif key == ord('c'):
                _info("开始校准...")
                calibration = run_calibration(gaze_pipeline, cap, screen_w, screen_h, device)
            elif key == ord('n'):
                pid = problem_id if problem_id else f"unknown_{collector.task_count + 1}"
                diff = DIFFICULTIES[collector.current_difficulty_idx]
                collector.start_task(pid, diff)
            elif key == ord('1'):
                collector.current_difficulty_idx = 0
            elif key == ord('2'):
                collector.current_difficulty_idx = 1
            elif key == ord('3'):
                collector.current_difficulty_idx = 2
            elif key == ord('4'):
                collector.current_difficulty_idx = 3
            elif key == ord('a') and collector.current_task:
                collector.end_task("AC", screen_w, screen_h)
            elif key == ord('w') and collector.current_task:
                collector.end_task("WA", screen_w, screen_h)
            elif key == ord('t') and collector.current_task:
                collector.end_task("TLE", screen_w, screen_h)
            elif key == ord('g') and collector.current_task:
                collector.end_task("GIVEUP", screen_w, screen_h)
            elif key == ord('p'):
                collector.recording = not collector.recording
                _info(f"Recording: {'ON' if collector.recording else 'OFF'}")
            elif key == ord('b'):
                # 手动触发一次眨眼事件（用于测试眨眼指标写入）
                collector.add_manual_blink(time.time())
            elif key == ord('+') or key == ord('='):
                # 增加平滑度 (降低灵敏度)
                filter_x.mincutoff = max(0.1, filter_x.mincutoff - 0.1)
                filter_y.mincutoff = max(0.1, filter_y.mincutoff - 0.1)
                _info(f"Smoothing: {filter_x.mincutoff:.1f} (MORE smooth)")
            elif key == ord('-'):
                # 降低平滑度 (提高灵敏度)
                filter_x.mincutoff = min(2.0, filter_x.mincutoff + 0.1)
                filter_y.mincutoff = min(2.0, filter_y.mincutoff + 0.1)
                _info(f"Smoothing: {filter_x.mincutoff:.1f} (LESS smooth)")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        _info("Stopping...")
        aoi_server.stop()
        _info("Saving data...")
        try:
            collector.export_data()
            _info(f"数据已保存: cognitive_data/{collector.session_id}/")
            _debug(f"  - Gaze records: {len(collector.session_data.gaze_records)}")
            _debug(f"  - Fixations: {len(collector.session_data.fixations)}")
            _debug(f"  - Blinks: {len(collector.session_data.blinks)}")
            _debug(f"  - AOI transitions: {len(collector.session_data.transitions)}")
            _debug(f"  - Tasks: {len(collector.session_data.tasks)}")
        except Exception as e:
            _warn(f"Error saving data: {e}")
            import traceback
            traceback.print_exc()
        cap.release()
        cv2.destroyAllWindows()
        _info("Done.")


if __name__ == "__main__":
    main()

