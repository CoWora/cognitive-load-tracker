"""
AOI 数据分析脚本

功能：
1. 计算各 AOI 区域的注视指标
2. 分析 AOI 转换模式
3. 计算认知负荷相关特征
4. 生成分析报告

输出指标：
- 各区域注视时间、次数、占比
- 区域转换矩阵
- 转换熵
- 认知负荷指数估计
"""

import pandas as pd
import numpy as np
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class AOIAnalyzer:
    """AOI 数据分析器"""
    
    def __init__(self, session_dir: str):
        self.session_dir = session_dir
        self.gaze_data = None
        self.fixations = None
        self.transitions = None
        self.events = None
        self.meta = None
        
        self._load_data()
    
    def _load_data(self):
        """加载数据"""
        # 加载视线数据
        gaze_file = os.path.join(self.session_dir, "gaze_data.csv")
        if os.path.exists(gaze_file):
            self.gaze_data = pd.read_csv(gaze_file)
            print(f"加载视线数据: {len(self.gaze_data)} 条记录")
        
        # 加载注视数据
        fix_file = os.path.join(self.session_dir, "fixations.csv")
        if os.path.exists(fix_file):
            self.fixations = pd.read_csv(fix_file)
            print(f"加载注视数据: {len(self.fixations)} 条记录")
        
        # 加载转换数据
        trans_file = os.path.join(self.session_dir, "aoi_transitions.csv")
        if os.path.exists(trans_file):
            self.transitions = pd.read_csv(trans_file)
            print(f"加载转换数据: {len(self.transitions)} 条记录")
        
        # 加载事件数据
        event_file = os.path.join(self.session_dir, "events.csv")
        if os.path.exists(event_file):
            self.events = pd.read_csv(event_file)
            print(f"加载事件数据: {len(self.events)} 条记录")
        
        # 加载元数据
        meta_file = os.path.join(self.session_dir, "session_meta.json")
        if os.path.exists(meta_file):
            with open(meta_file, 'r', encoding='utf-8') as f:
                self.meta = json.load(f)
    
    def calculate_aoi_metrics(self, task_id: str = None) -> Dict:
        """
        计算各 AOI 区域的指标
        
        Args:
            task_id: 任务ID，None 表示全部数据
        
        Returns:
            各区域的指标字典
        """
        if self.gaze_data is None:
            return {}
        
        data = self.gaze_data
        if task_id:
            data = data[data['task_id'] == task_id]
        
        if len(data) == 0:
            return {}
        
        # 计算总时长（基于时间戳）
        total_duration = data['timestamp'].max() - data['timestamp'].min()
        total_records = len(data)
        
        metrics = {}
        
        for aoi in data['aoi_region'].unique():
            aoi_data = data[data['aoi_region'] == aoi]
            aoi_name = aoi_data['aoi_name'].iloc[0] if len(aoi_data) > 0 else aoi
            
            # 基于记录数估算时间（假设 30fps）
            record_count = len(aoi_data)
            estimated_time = record_count / 30.0  # 秒
            
            # 注视次数（从 fixations 数据）
            if self.fixations is not None:
                fix_data = self.fixations[self.fixations['aoi_region'] == aoi]
                if task_id:
                    fix_data = fix_data[fix_data['task_id'] == task_id]
                fixation_count = len(fix_data)
                avg_fixation_duration = fix_data['duration'].mean() if len(fix_data) > 0 else 0
                total_fixation_time = fix_data['duration'].sum() if len(fix_data) > 0 else 0
            else:
                fixation_count = 0
                avg_fixation_duration = 0
                total_fixation_time = 0
            
            metrics[aoi] = {
                'name': aoi_name,
                'record_count': record_count,
                'record_percentage': record_count / total_records * 100,
                'estimated_time': estimated_time,
                'time_percentage': estimated_time / total_duration * 100 if total_duration > 0 else 0,
                'fixation_count': fixation_count,
                'avg_fixation_duration': avg_fixation_duration * 1000,  # 转换为毫秒
                'total_fixation_time': total_fixation_time
            }
        
        return metrics
    
    def calculate_transition_matrix(self, task_id: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        计算 AOI 转换矩阵
        
        Returns:
            (转换矩阵 DataFrame, 转换统计)
        """
        if self.transitions is None or len(self.transitions) == 0:
            return pd.DataFrame(), {}
        
        data = self.transitions
        if task_id:
            data = data[data['task_id'] == task_id]
        
        if len(data) == 0:
            return pd.DataFrame(), {}
        
        # 统计转换次数
        transition_counts = defaultdict(lambda: defaultdict(int))
        for _, row in data.iterrows():
            transition_counts[row['from_aoi']][row['to_aoi']] += 1
        
        # 获取所有 AOI
        all_aois = set(data['from_aoi'].unique()) | set(data['to_aoi'].unique())
        all_aois = sorted(all_aois)
        
        # 创建矩阵
        matrix = pd.DataFrame(0, index=all_aois, columns=all_aois)
        for from_aoi in transition_counts:
            for to_aoi in transition_counts[from_aoi]:
                matrix.loc[from_aoi, to_aoi] = transition_counts[from_aoi][to_aoi]
        
        # 计算关键转换统计
        stats = {
            'total_transitions': len(data),
            'unique_transitions': len(set(zip(data['from_aoi'], data['to_aoi']))),
            'most_frequent': None,
            'problem_to_code': 0,
            'code_to_problem': 0,
            'example_to_code': 0
        }
        
        # 找出最频繁的转换
        if len(data) > 0:
            trans_pairs = list(zip(data['from_aoi'], data['to_aoi']))
            most_common = Counter(trans_pairs).most_common(1)
            if most_common:
                stats['most_frequent'] = {
                    'from': most_common[0][0][0],
                    'to': most_common[0][0][1],
                    'count': most_common[0][1]
                }
        
        # 计算关键转换
        for _, row in data.iterrows():
            if 'PROBLEM' in row['from_aoi'] and 'CODE' in row['to_aoi']:
                stats['problem_to_code'] += 1
            elif 'CODE' in row['from_aoi'] and 'PROBLEM' in row['to_aoi']:
                stats['code_to_problem'] += 1
            elif 'EXAMPLE' in row['from_aoi'] and 'CODE' in row['to_aoi']:
                stats['example_to_code'] += 1
        
        return matrix, stats
    
    def calculate_transition_entropy(self, task_id: str = None) -> float:
        """
        计算转换熵（衡量视线模式的混乱程度）
        
        高熵 = 视线模式混乱，认知负荷高
        低熵 = 视线模式有序，认知负荷低
        """
        if self.transitions is None or len(self.transitions) == 0:
            return 0.0
        
        data = self.transitions
        if task_id:
            data = data[data['task_id'] == task_id]
        
        if len(data) == 0:
            return 0.0
        
        # 统计转换概率
        trans_pairs = list(zip(data['from_aoi'], data['to_aoi']))
        counts = Counter(trans_pairs)
        total = sum(counts.values())
        
        # 计算熵
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def calculate_cognitive_load_index(self, task_id: str = None) -> Dict:
        """
        计算认知负荷指数
        
        基于以下指标综合计算：
        1. 题目-代码转换频率（越高负荷越高）
        2. 转换熵（越高负荷越高）
        3. 代码区注视占比（越高负荷越低）
        4. 平均注视时长（越长负荷越高）
        """
        metrics = self.calculate_aoi_metrics(task_id)
        _, trans_stats = self.calculate_transition_matrix(task_id)
        entropy = self.calculate_transition_entropy(task_id)
        
        if not metrics or not trans_stats:
            return {'index': 0, 'level': 'N/A', 'factors': {}}
        
        # 获取总时长
        if self.gaze_data is not None:
            data = self.gaze_data
            if task_id:
                data = data[data['task_id'] == task_id]
            duration = data['timestamp'].max() - data['timestamp'].min()
        else:
            duration = 1
        
        # 计算各因素
        factors = {}
        
        # 1. 题目-代码切换频率（次/分钟）
        switch_count = trans_stats.get('problem_to_code', 0) + trans_stats.get('code_to_problem', 0)
        switch_rate = switch_count / (duration / 60) if duration > 0 else 0
        factors['switch_rate'] = switch_rate
        
        # 2. 转换熵
        factors['transition_entropy'] = entropy
        
        # 3. 代码区注视占比
        code_percentage = 0
        for aoi, m in metrics.items():
            if 'CODE' in aoi:
                code_percentage = m.get('time_percentage', 0)
                break
        factors['code_focus'] = code_percentage
        
        # 4. 平均注视时长
        if self.fixations is not None:
            fix_data = self.fixations
            if task_id:
                fix_data = fix_data[fix_data['task_id'] == task_id]
            avg_fix_duration = fix_data['duration'].mean() * 1000 if len(fix_data) > 0 else 0
        else:
            avg_fix_duration = 0
        factors['avg_fixation_duration'] = avg_fix_duration
        
        # 综合计算认知负荷指数 (0-100)
        # 归一化各因素后加权
        index = 0
        
        # 切换频率：0-10次/分钟 → 0-30分
        index += min(switch_rate / 10, 1) * 30
        
        # 转换熵：0-4 bits → 0-25分
        index += min(entropy / 4, 1) * 25
        
        # 代码区占比：100%-0% → 0-25分（占比越低负荷越高）
        index += (1 - code_percentage / 100) * 25
        
        # 平均注视时长：0-500ms → 0-20分
        index += min(avg_fix_duration / 500, 1) * 20
        
        # 确定负荷等级
        if index < 25:
            level = "低 (Low)"
        elif index < 50:
            level = "中 (Medium)"
        elif index < 75:
            level = "高 (High)"
        else:
            level = "极高 (Very High)"
        
        return {
            'index': round(index, 2),
            'level': level,
            'factors': factors
        }
    
    def generate_report(self, output_file: str = None) -> str:
        """生成分析报告"""
        report = []
        report.append("=" * 70)
        report.append("认知负荷数据分析报告")
        report.append("=" * 70)
        
        if self.meta:
            report.append(f"\n会话ID: {self.meta.get('session_id', 'N/A')}")
            report.append(f"总时长: {self.meta.get('duration', 0):.2f} 秒")
            report.append(f"视线记录: {self.meta.get('total_gaze_records', 0)} 条")
            report.append(f"注视次数: {self.meta.get('total_fixations', 0)} 次")
            report.append(f"AOI转换: {self.meta.get('total_transitions', 0)} 次")
        
        # 获取任务列表
        tasks = ['ALL']
        if self.gaze_data is not None:
            tasks.extend(self.gaze_data['task_id'].unique().tolist())
        
        for task_id in tasks:
            task_filter = None if task_id == 'ALL' else task_id
            
            report.append(f"\n{'=' * 70}")
            report.append(f"任务: {task_id}")
            report.append("=" * 70)
            
            # AOI 指标
            report.append("\n【各区域注视指标】")
            report.append("-" * 50)
            metrics = self.calculate_aoi_metrics(task_filter)
            for aoi, m in sorted(metrics.items()):
                report.append(f"\n  {aoi} ({m['name']}):")
                report.append(f"    注视占比: {m['time_percentage']:.1f}%")
                report.append(f"    注视次数: {m['fixation_count']} 次")
                report.append(f"    平均注视时长: {m['avg_fixation_duration']:.1f} ms")
            
            # 转换分析
            report.append("\n\n【AOI 转换分析】")
            report.append("-" * 50)
            matrix, trans_stats = self.calculate_transition_matrix(task_filter)
            report.append(f"  总转换次数: {trans_stats.get('total_transitions', 0)}")
            report.append(f"  题目→代码: {trans_stats.get('problem_to_code', 0)} 次")
            report.append(f"  代码→题目: {trans_stats.get('code_to_problem', 0)} 次")
            report.append(f"  示例→代码: {trans_stats.get('example_to_code', 0)} 次")
            
            if trans_stats.get('most_frequent'):
                mf = trans_stats['most_frequent']
                report.append(f"  最频繁转换: {mf['from']} → {mf['to']} ({mf['count']} 次)")
            
            # 认知负荷指数
            report.append("\n\n【认知负荷评估】")
            report.append("-" * 50)
            cl = self.calculate_cognitive_load_index(task_filter)
            report.append(f"  认知负荷指数: {cl['index']}/100")
            report.append(f"  负荷等级: {cl['level']}")
            report.append("\n  影响因素:")
            report.append(f"    题目-代码切换频率: {cl['factors'].get('switch_rate', 0):.2f} 次/分钟")
            report.append(f"    转换熵: {cl['factors'].get('transition_entropy', 0):.2f} bits")
            report.append(f"    代码区专注度: {cl['factors'].get('code_focus', 0):.1f}%")
            report.append(f"    平均注视时长: {cl['factors'].get('avg_fixation_duration', 0):.1f} ms")
        
        report.append("\n" + "=" * 70)
        report.append("报告生成完成")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        # 保存报告
        if output_file is None:
            output_file = os.path.join(self.session_dir, "analysis_report.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n报告已保存: {output_file}")
        
        return report_text
    
    def plot_aoi_distribution(self, task_id: str = None, save_path: str = None):
        """绘制 AOI 分布图"""
        metrics = self.calculate_aoi_metrics(task_id)
        if not metrics:
            print("无数据可绘制")
            return
        
        # 准备数据
        labels = [f"{aoi}\n({m['name']})" for aoi, m in metrics.items()]
        sizes = [m['time_percentage'] for m in metrics.values()]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 饼图
        axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0].set_title('AOI 注视时间分布')
        
        # 柱状图
        x = range(len(labels))
        bars = axes[1].bar(x, sizes, color=colors)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([m['name'] for m in metrics.values()], rotation=45, ha='right')
        axes[1].set_ylabel('注视占比 (%)')
        axes[1].set_title('AOI 注视时间对比')
        
        # 添加数值标签
        for bar, size in zip(bars, sizes):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{size:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.session_dir, "aoi_distribution.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()
    
    def plot_transition_heatmap(self, task_id: str = None, save_path: str = None):
        """绘制转换矩阵热力图"""
        matrix, _ = self.calculate_transition_matrix(task_id)
        if matrix.empty:
            print("无转换数据可绘制")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(matrix.values, cmap='YlOrRd')
        
        # 设置标签
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_yticks(range(len(matrix.index)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(matrix.index)
        
        # 添加数值标签
        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                value = matrix.iloc[i, j]
                if value > 0:
                    text_color = 'white' if value > matrix.values.max() / 2 else 'black'
                    ax.text(j, i, str(int(value)), ha='center', va='center', color=text_color)
        
        ax.set_xlabel('To AOI')
        ax.set_ylabel('From AOI')
        ax.set_title('AOI 转换矩阵')
        
        plt.colorbar(im, ax=ax, label='转换次数')
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.session_dir, "transition_heatmap.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()


def analyze_session(session_dir: str):
    """分析单个会话"""
    print(f"\n分析会话: {session_dir}")
    
    analyzer = AOIAnalyzer(session_dir)
    
    # 生成报告
    report = analyzer.generate_report()
    print(report)
    
    # 生成图表
    try:
        analyzer.plot_aoi_distribution()
        analyzer.plot_transition_heatmap()
    except Exception as e:
        print(f"图表生成失败: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        session_dir = sys.argv[1]
    else:
        # 查找最新的会话目录
        data_dir = "cognitive_data"
        if os.path.exists(data_dir):
            sessions = [d for d in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, d))]
            if sessions:
                session_dir = os.path.join(data_dir, sorted(sessions)[-1])
            else:
                print("未找到会话数据")
                exit(1)
        else:
            print(f"数据目录不存在: {data_dir}")
            exit(1)
    
    analyze_session(session_dir)

