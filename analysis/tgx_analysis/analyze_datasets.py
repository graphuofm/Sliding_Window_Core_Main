import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_dataset(file_path, dataset_name, sample_size=None):
    """加载数据集"""
    print(f"\n{'='*60}")
    print(f"加载数据集: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # 读取数据
        print(f"读取文件: {file_path}")
        
        # 对于大文件，可以只读取一部分
        if sample_size and 'social_media_100M' in dataset_name:
            print(f"由于文件较大，只读取前 {sample_size} 行进行分析...")
            df = pd.read_csv(file_path, sep=' ', header=None, 
                           names=['source', 'target', 'timestamp'],
                           nrows=sample_size)
        else:
            df = pd.read_csv(file_path, sep=' ', header=None, 
                           names=['source', 'target', 'timestamp'])
        
        print(f"数据加载完成！共 {len(df):,} 条边")
        return df
    
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def analyze_temporal_graph(df, dataset_name):
    """分析时序图的统计特性"""
    print(f"\n分析 {dataset_name} 的统计特性...")
    
    # 基本统计
    stats = {}
    
    # 1. 节点统计
    all_nodes = pd.concat([df['source'], df['target']]).unique()
    stats['num_nodes'] = len(all_nodes)
    stats['num_edges'] = len(df)
    
    # 2. 时间统计
    stats['min_timestamp'] = df['timestamp'].min()
    stats['max_timestamp'] = df['timestamp'].max()
    stats['time_span'] = stats['max_timestamp'] - stats['min_timestamp']
    
    # 3. 度统计
    source_counts = df['source'].value_counts()
    target_counts = df['target'].value_counts()
    
    # 出度统计
    stats['avg_out_degree'] = source_counts.mean()
    stats['max_out_degree'] = source_counts.max()
    stats['min_out_degree'] = source_counts.min()
    
    # 入度统计
    stats['avg_in_degree'] = target_counts.mean()
    stats['max_in_degree'] = target_counts.max()
    stats['min_in_degree'] = target_counts.min()
    
    # 4. 时间分布统计
    unique_timestamps = df['timestamp'].nunique()
    stats['unique_timestamps'] = unique_timestamps
    
    # 打印统计结果
    print("\n基本统计信息:")
    print(f"- 节点数: {stats['num_nodes']:,}")
    print(f"- 边数: {stats['num_edges']:,}")
    print(f"- 唯一时间戳数: {stats['unique_timestamps']:,}")
    print(f"- 时间跨度: {stats['time_span']} 秒")
    print(f"\n度分布统计:")
    print(f"- 平均出度: {stats['avg_out_degree']:.2f}")
    print(f"- 最大出度: {stats['max_out_degree']:,}")
    print(f"- 平均入度: {stats['avg_in_degree']:.2f}")
    print(f"- 最大入度: {stats['max_in_degree']:,}")
    
    return stats, source_counts, target_counts

def create_visualizations(df, dataset_name, source_counts, target_counts):
    """创建可视化图表"""
    print(f"\n为 {dataset_name} 创建可视化...")
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. 度分布图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 出度分布
    out_degree_dist = source_counts.value_counts().sort_index()
    ax1.loglog(out_degree_dist.index, out_degree_dist.values, 'bo-', alpha=0.7)
    ax1.set_xlabel('出度')
    ax1.set_ylabel('节点数')
    ax1.set_title('出度分布 (log-log scale)')
    ax1.grid(True, alpha=0.3)
    
    # 入度分布
    in_degree_dist = target_counts.value_counts().sort_index()
    ax2.loglog(in_degree_dist.index, in_degree_dist.values, 'ro-', alpha=0.7)
    ax2.set_xlabel('入度')
    ax2.set_ylabel('节点数')
    ax2.set_title('入度分布 (log-log scale)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{dataset_name} - 度分布分析')
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_degree_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 时间分布图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 按时间戳统计边数
    time_dist = df['timestamp'].value_counts().sort_index()
    
    # 如果时间戳太多，进行聚合
    if len(time_dist) > 1000:
        # 转换为时间序列并按小时聚合
        time_series = pd.Series(time_dist.values, index=pd.to_datetime(time_dist.index, unit='s'))
        time_series_hourly = time_series.resample('H').sum()
        time_series_hourly.plot(ax=ax, alpha=0.7)
        ax.set_xlabel('时间')
        ax.set_ylabel('每小时边数')
        ax.set_title(f'{dataset_name} - 时间分布（按小时聚合）')
    else:
        ax.bar(range(len(time_dist)), time_dist.values, alpha=0.7)
        ax.set_xlabel('时间戳索引')
        ax.set_ylabel('边数')
        ax.set_title(f'{dataset_name} - 时间分布')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_temporal_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 节点活跃度热图（Top 20 节点）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 最活跃的源节点
    top_sources = source_counts.head(20)
    ax1.barh(range(len(top_sources)), top_sources.values, alpha=0.7)
    ax1.set_yticks(range(len(top_sources)))
    ax1.set_yticklabels([f'Node {i}' for i in top_sources.index])
    ax1.set_xlabel('出度')
    ax1.set_title('Top 20 最活跃源节点')
    ax1.grid(True, alpha=0.3)
    
    # 最活跃的目标节点
    top_targets = target_counts.head(20)
    ax2.barh(range(len(top_targets)), top_targets.values, alpha=0.7, color='red')
    ax2.set_yticks(range(len(top_targets)))
    ax2.set_yticklabels([f'Node {i}' for i in top_targets.index])
    ax2.set_xlabel('入度')
    ax2.set_title('Top 20 最受欢迎目标节点')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{dataset_name} - 节点活跃度分析')
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_node_activity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存!")

def save_summary_report(stats_dict, output_file='analysis_summary.txt'):
    """保存分析报告"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("时序图数据集分析报告\n")
        f.write("="*60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for dataset_name, stats in stats_dict.items():
            f.write(f"\n{dataset_name}:\n")
            f.write("-"*40 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

def main():
    """主函数"""
    # 数据集路径
    dataset_path = os.path.expanduser("~/dataset/")
    
    # 要分析的数据集
    datasets = {
        'citation_network_10M': 'citation_network_10M.txt',
        'social_media_100M': 'social_media_100M.txt'
    }
    
    # 存储所有统计结果
    all_stats = {}
    
    for name, filename in datasets.items():
        file_path = os.path.join(dataset_path, filename)
        
        # 对于100M的数据集，可以只分析一部分
        if '100M' in name:
            # 只读取前1000万行进行分析
            df = load_dataset(file_path, name, sample_size=10000000)
        else:
            df = load_dataset(file_path, name)
        
        if df is not None:
            # 分析统计特性
            stats, source_counts, target_counts = analyze_temporal_graph(df, name)
            all_stats[name] = stats
            
            # 创建可视化
            create_visualizations(df, name, source_counts, target_counts)
            
            # 清理内存
            del df
            print(f"\n{name} 分析完成！")
    
    # 保存汇总报告
    save_summary_report(all_stats)
    print("\n所有分析完成！结果已保存。")
    
    # 列出生成的文件
    print("\n生成的文件:")
    for file in os.listdir('.'):
        if file.endswith(('.png', '.txt')):
            print(f"- {file}")

if __name__ == "__main__":
    main()