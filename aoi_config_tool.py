"""
AOI 区域配置工具

功能：
1. 截取洛谷页面截图
2. 可视化标注 AOI 区域
3. 保存配置到 JSON 文件

使用方法：
1. 运行此工具
2. 打开洛谷题目页面
3. 按 S 截图
4. 用鼠标拖拽标注各个区域
5. 按数字键选择区域类型
6. 按 Enter 保存配置
"""

import cv2
import numpy as np
import json
import os
import ctypes
from datetime import datetime

try:
    import pyautogui
except ImportError:
    print("请安装 pyautogui: pip install pyautogui")
    exit(1)


# 默认 AOI 区域定义
AOI_DEFINITIONS = {
    "1": {"id": "A_TITLE", "name": "题目标题", "color": (255, 100, 100)},
    "2": {"id": "B_PROBLEM", "name": "题目描述", "color": (100, 255, 100)},
    "3": {"id": "C_IO_FORMAT", "name": "输入输出格式", "color": (100, 100, 255)},
    "4": {"id": "D_EXAMPLES", "name": "示例", "color": (255, 255, 100)},
    "5": {"id": "E_CONSTRAINTS", "name": "约束/提示", "color": (255, 100, 255)},
}


class AOIConfigTool:
    """AOI 配置工具"""
    
    def __init__(self):
        self.screenshot = None
        self.display_image = None
        self.regions = {}
        self.current_region = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.selected_type = "1"  # 默认选择标题
        
        # 屏幕尺寸
        try:
            user32 = ctypes.windll.user32
            self.screen_w = user32.GetSystemMetrics(0)
            self.screen_h = user32.GetSystemMetrics(1)
        except:
            self.screen_w, self.screen_h = 1920, 1080
        
        self.window_name = "AOI Config Tool"
    
    def take_screenshot(self):
        """截取屏幕"""
        print("3秒后截图，请切换到洛谷页面...")
        cv2.waitKey(3000)
        
        self.screenshot = pyautogui.screenshot()
        self.screenshot = np.array(self.screenshot)
        self.screenshot = cv2.cvtColor(self.screenshot, cv2.COLOR_RGB2BGR)
        
        # 缩放以适应显示
        scale = min(1.0, 1200 / self.screen_w, 800 / self.screen_h)
        new_w = int(self.screen_w * scale)
        new_h = int(self.screen_h * scale)
        self.display_scale = scale
        self.display_image = cv2.resize(self.screenshot, (new_w, new_h))
        
        print(f"截图完成: {self.screen_w}x{self.screen_h}")
        print(f"显示尺寸: {new_w}x{new_h} (缩放: {scale:.2f})")
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调"""
        if self.display_image is None:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
            if self.start_point and self.end_point:
                # 转换为归一化坐标
                x1 = min(self.start_point[0], self.end_point[0]) / self.display_image.shape[1]
                y1 = min(self.start_point[1], self.end_point[1]) / self.display_image.shape[0]
                x2 = max(self.start_point[0], self.end_point[0]) / self.display_image.shape[1]
                y2 = max(self.start_point[1], self.end_point[1]) / self.display_image.shape[0]
                
                if self.selected_type in AOI_DEFINITIONS:
                    aoi_def = AOI_DEFINITIONS[self.selected_type]
                    self.regions[aoi_def["id"]] = {
                        "name": aoi_def["name"],
                        "x1": round(x1, 4),
                        "y1": round(y1, 4),
                        "x2": round(x2, 4),
                        "y2": round(y2, 4),
                        "color": aoi_def["color"]
                    }
                    print(f"已标注: {aoi_def['name']} ({x1:.2f}, {y1:.2f}) - ({x2:.2f}, {y2:.2f})")
    
    def draw_regions(self, img):
        """绘制已标注的区域"""
        result = img.copy()
        h, w = result.shape[:2]
        
        for aoi_id, region in self.regions.items():
            x1 = int(region["x1"] * w)
            y1 = int(region["y1"] * h)
            x2 = int(region["x2"] * w)
            y2 = int(region["y2"] * h)
            color = region.get("color", (255, 255, 255))
            
            # 半透明填充
            overlay = result.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
            
            # 边框
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # 标签
            label = f"{aoi_id}: {region['name']}"
            cv2.putText(result, label, (x1 + 5, y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 绘制当前正在画的矩形
        if self.drawing and self.start_point and self.end_point:
            cv2.rectangle(result, self.start_point, self.end_point, (0, 255, 255), 2)
        
        return result
    
    def draw_ui(self, img):
        """绘制 UI"""
        result = img.copy()
        h, w = result.shape[:2]
        
        # 顶部信息栏
        cv2.rectangle(result, (0, 0), (w, 80), (40, 40, 40), -1)
        
        # 标题
        cv2.putText(result, "AOI Configuration Tool", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 当前选择
        if self.selected_type in AOI_DEFINITIONS:
            aoi_def = AOI_DEFINITIONS[self.selected_type]
            cv2.putText(result, f"Current: [{self.selected_type}] {aoi_def['name']}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, aoi_def['color'], 1)
        
        # 快捷键提示
        cv2.putText(result, "Keys: 1-5 Select region | S Screenshot | Enter Save | Q Quit",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # 区域列表
        y = 100
        cv2.putText(result, "Regions:", (w - 200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for key, aoi_def in AOI_DEFINITIONS.items():
            y += 20
            status = "✓" if aoi_def["id"] in self.regions else " "
            text = f"[{key}] {status} {aoi_def['name']}"
            color = aoi_def['color'] if aoi_def["id"] in self.regions else (150, 150, 150)
            cv2.putText(result, text, (w - 200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return result
    
    def save_config(self, filename="aoi_config.json"):
        """保存配置"""
        # 添加代码区默认配置
        config = self.regions.copy()
        config["F_CODE_EDITOR"] = {
            "name": "代码区",
            "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0,
            "color": [100, 255, 255]
        }
        
        # 转换颜色为列表（JSON 兼容）
        for key in config:
            if isinstance(config[key].get("color"), tuple):
                config[key]["color"] = list(config[key]["color"])
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"\n配置已保存: {filename}")
        print("配置内容:")
        for aoi_id, region in config.items():
            print(f"  {aoi_id}: {region['name']} ({region['x1']:.2f}, {region['y1']:.2f}) - ({region['x2']:.2f}, {region['y2']:.2f})")
    
    def run(self):
        """运行配置工具"""
        print("=" * 60)
        print("AOI 区域配置工具")
        print("=" * 60)
        print("\n操作说明:")
        print("  1. 按 S 截取屏幕（3秒倒计时）")
        print("  2. 按 1-5 选择要标注的区域类型")
        print("  3. 用鼠标拖拽标注区域")
        print("  4. 按 Enter 保存配置")
        print("  5. 按 Q 退出")
        print("\n区域类型:")
        for key, aoi_def in AOI_DEFINITIONS.items():
            print(f"  [{key}] {aoi_def['name']}")
        print("=" * 60)
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # 初始黑色画面
        blank = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(blank, "Press S to take screenshot", (200, 300),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        while True:
            if self.display_image is not None:
                display = self.draw_regions(self.display_image)
                display = self.draw_ui(display)
            else:
                display = blank
            
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.take_screenshot()
            elif key == 13:  # Enter
                if self.regions:
                    self.save_config()
                else:
                    print("请先标注区域！")
            elif chr(key) in AOI_DEFINITIONS:
                self.selected_type = chr(key)
                aoi_def = AOI_DEFINITIONS[self.selected_type]
                print(f"选择区域: [{self.selected_type}] {aoi_def['name']}")
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tool = AOIConfigTool()
    tool.run()

