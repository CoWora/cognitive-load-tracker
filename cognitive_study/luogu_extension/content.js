/**
 * Luogu AOI Tracker - Content Script (HTTP 版本)
 * 
 * 使用 HTTP POST 代替 WebSocket 来绕过 CSP 限制
 */

(function() {
    'use strict';

    const API_URL = 'http://127.0.0.1:8766/aoi';
    let connected = false;
    let lastProblemId = null;

    function getElementScreenRect(element) {
        if (!element) return null;
        const rect = element.getBoundingClientRect();
        // 计算元素在屏幕上的绝对位置
        const screenX = window.screenX + (window.outerWidth - window.innerWidth) / 2;
        const screenY = window.screenY + (window.outerHeight - window.innerHeight);
        return {
            x1: Math.round(rect.left + screenX),
            y1: Math.round(rect.top + screenY),
            x2: Math.round(rect.right + screenX),
            y2: Math.round(rect.bottom + screenY),
            width: Math.round(rect.width),
            height: Math.round(rect.height)
        };
    }

    function extractProblemId() {
        // 从 URL 提取
        const urlMatch = window.location.pathname.match(/\/problem\/(P\d+|CF\d+[A-Z]?|AT_\w+|SP\w+|UVA\d+|B\d+)/i);
        if (urlMatch) return urlMatch[1].toUpperCase();

        // 从标题提取
        const titleMatch = document.title.match(/(P\d+|CF\d+[A-Z]?|AT_\w+|SP\w+|UVA\d+|B\d+)/i);
        if (titleMatch) return titleMatch[1].toUpperCase();

        return null;
    }

    function extractProblemTitle() {
        const h1 = document.querySelector('h1');
        return h1 ? h1.textContent.trim() : '';
    }

    function findProblemContent() {
        // 尝试找到题目主体区域
        // 洛谷通常用 .problem-card 或类似的容器
        const selectors = [
            '.problem-card',
            '.card',
            '[class*="problem"]',
            'article',
            'main',
            '.content'
        ];
        
        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el && el.offsetHeight > 200) {
                return el;
            }
        }
        return null;
    }

    function collectAOIData() {
        const isProblemPage = window.location.pathname.includes('/problem/');
        
        const data = {
            timestamp: Date.now(),
            url: window.location.href,
            is_problem_page: isProblemPage,
            problem_id: extractProblemId(),
            problem_title: extractProblemTitle(),
            screen_width: window.screen.width,
            screen_height: window.screen.height,
            window_x: window.screenX,
            window_y: window.screenY,
            viewport_width: window.innerWidth,
            viewport_height: window.innerHeight,
            scroll_x: window.scrollX,
            scroll_y: window.scrollY,
            aoi_regions: {}
        };

        if (!isProblemPage) {
            return data;
        }

        // 方法: 把整个页面内容区域作为一个大的 AOI
        // 然后根据垂直位置划分区域
        
        const problemContent = findProblemContent();
        if (problemContent) {
            const contentRect = getElementScreenRect(problemContent);
            if (contentRect) {
                // 将内容区域垂直划分
                const totalHeight = contentRect.y2 - contentRect.y1;
                
                // 横向大幅扩展边界 (几乎覆盖整个屏幕宽度)
                const expandX = 500;
                const x1 = Math.max(0, contentRect.x1 - expandX);
                const x2 = Math.min(window.screen.width, contentRect.x2 + expandX);
                
                // 标题区：顶部 15% (扩大)
                data.aoi_regions['A_TITLE'] = {
                    found: true,
                    x1: x1,
                    y1: contentRect.y1,
                    x2: x2,
                    y2: contentRect.y1 + Math.round(totalHeight * 0.15)
                };
                
                // 题目描述：15% - 50%
                data.aoi_regions['B_PROBLEM'] = {
                    found: true,
                    x1: x1,
                    y1: contentRect.y1 + Math.round(totalHeight * 0.15),
                    x2: x2,
                    y2: contentRect.y1 + Math.round(totalHeight * 0.50)
                };
                
                // 输入输出格式：50% - 65%
                data.aoi_regions['C_IO_FORMAT'] = {
                    found: true,
                    x1: x1,
                    y1: contentRect.y1 + Math.round(totalHeight * 0.50),
                    x2: x2,
                    y2: contentRect.y1 + Math.round(totalHeight * 0.65)
                };
                
                // 示例：65% - 85%
                data.aoi_regions['D_EXAMPLES'] = {
                    found: true,
                    x1: x1,
                    y1: contentRect.y1 + Math.round(totalHeight * 0.65),
                    x2: x2,
                    y2: contentRect.y1 + Math.round(totalHeight * 0.85)
                };
                
                // 提示/约束：85% - 100%
                data.aoi_regions['E_CONSTRAINTS'] = {
                    found: true,
                    x1: x1,
                    y1: contentRect.y1 + Math.round(totalHeight * 0.85),
                    x2: x2,
                    y2: contentRect.y2
                };
            }
        } else {
            // 备用方案：使用整个视口
            const screenX = window.screenX + (window.outerWidth - window.innerWidth) / 2;
            const screenY = window.screenY + (window.outerHeight - window.innerHeight);
            
            const vpWidth = window.innerWidth;
            const vpHeight = window.innerHeight;
            
            // 标题区：顶部栏
            data.aoi_regions['A_TITLE'] = {
                found: true,
                x1: screenX,
                y1: screenY,
                x2: screenX + vpWidth,
                y2: screenY + 100
            };
            
            // 题目区：中间主体
            data.aoi_regions['B_PROBLEM'] = {
                found: true,
                x1: screenX,
                y1: screenY + 100,
                x2: screenX + vpWidth,
                y2: screenY + vpHeight
            };
        }

        return data;
    }

    // 创建状态指示器
    function createIndicator() {
        const existing = document.getElementById('aoi-tracker-indicator');
        if (existing) existing.remove();

        const indicator = document.createElement('div');
        indicator.id = 'aoi-tracker-indicator';
        indicator.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 6px 12px;
            background: rgba(0, 0, 0, 0.85);
            color: white;
            border-radius: 15px;
            font-size: 11px;
            z-index: 999999;
            font-family: monospace;
            display: flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        `;
        
        const dot = document.createElement('span');
        dot.id = 'aoi-dot';
        dot.style.cssText = `
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #ff4444;
            transition: background 0.3s;
        `;
        
        const text = document.createElement('span');
        text.id = 'aoi-text';
        text.textContent = 'AOI: ...';
        
        indicator.appendChild(dot);
        indicator.appendChild(text);
        document.body.appendChild(indicator);
    }

    function updateIndicator(isConnected, problemId) {
        const dot = document.getElementById('aoi-dot');
        const text = document.getElementById('aoi-text');
        if (!dot || !text) return;

        dot.style.background = isConnected ? '#44ff44' : '#ff4444';
        if (isConnected && problemId) {
            text.textContent = `AOI: ${problemId}`;
        } else if (isConnected) {
            text.textContent = 'AOI: OK';
        } else {
            text.textContent = 'AOI: X';
        }
    }

    // 发送数据到 Python 服务器
    async function sendData() {
        const data = collectAOIData();
        
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
                mode: 'cors'
            });
            
            if (response.ok) {
                if (!connected) {
                    connected = true;
                    console.log('[AOI Tracker] Connected to Python server');
                }
                updateIndicator(true, data.problem_id);
            } else {
                throw new Error('Response not ok');
            }
        } catch (e) {
            if (connected) {
                connected = false;
                console.log('[AOI Tracker] Disconnected, waiting to reconnect...');
            }
            updateIndicator(false, null);
        }
    }

    // 初始化
    function init() {
        console.log('[AOI Tracker] Init (HTTP mode)');
        createIndicator();
        
        // 立即发送一次
        sendData();
        
        // 每 200ms 发送一次数据
        setInterval(sendData, 200);

        // 监听滚动和大小变化
        window.addEventListener('scroll', sendData, { passive: true });
        window.addEventListener('resize', sendData);
        
        console.log('[AOI Tracker] Ready');
    }

    // 页面加载后初始化
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        setTimeout(init, 500);
    } else {
        window.addEventListener('DOMContentLoaded', () => setTimeout(init, 500));
    }
})();
