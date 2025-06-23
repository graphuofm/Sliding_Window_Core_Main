import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import tgx
    HAS_TGX = True
except ImportError:
    HAS_TGX = False
    print("警告: TGX 未安装，将使用基础分析")

def load_dataset(file_path, dataset_name, sample_size=None):
    """加载数据集"""
    print(f"\n{'='*60}")
    print(f"加载数据集: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        print(f"读取文件: {file_path}")
        
        if sample_size and '100M' in dataset_name:
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

def evaluate_temporal_patterns(df, dataset_name):
    """评估时序模式"""
    print(f"\n### 时序模式评估 ###")
    
    # 时间戳分析
    timestamps = df['timestamp'].values
    time_diffs = np.diff(np.sort(timestamps))
    
    # 时间间隔统计
    print(f"时间戳统计:")
    print(f"- 最小时间间隔: {time_diffs.min() if len(time_diffs) > 0 else 0}")
    print(f"- 最大时间间隔: {time_diffs.max() if len(time_diffs) > 0 else 0}")
    print(f"- 平均时间间隔: {time_diffs.mean() if len(time_diffs) > 0 else 0:.2f}")
    print(f"- 时间间隔标准差: {time_diffs.std() if len(time_diffs) > 0 else 0:.2f}")
    
    # 时间分布均匀性
    unique_timestamps = df['timestamp'].nunique()
    total_edges = len(df)
    temporal_density = unique_timestamps / total_edges
    print(f"\n时间分布:")
    print(f"- 时间密度: {temporal_density:.4f} (唯一时间戳/总边数)")
    print(f"- 平均每个时间戳的边数: {total_edges/unique_timestamps:.2f}")
    
    # 评估时间分布是否合理
    if temporal_density > 0.8:
        print("- 评估: 时间分布非常稀疏，适合模拟真实世界的时序网络")
    elif temporal_density > 0.5:
        print("- 评估: 时间分布较稀疏，可能适合某些时序任务")
    else:
        print("- 评估: 时间分布较密集，可能不够真实")
    
    return temporal_density

def evaluate_graph_structure(df, dataset_name):
    """评估图结构特性"""
    print(f"\n### 图结构评估 ###")
    
    # 基本统计
    all_nodes = pd.concat([df['source'], df['target']]).unique()
    num_nodes = len(all_nodes)
    num_edges = len(df)
    
    # 计算度分布
    source_counts = df['source'].value_counts()
    target_counts = df['target'].value_counts()
    
    # 度分布特征
    out_degrees = source_counts.values
    in_degrees = target_counts.values
    
    print(f"度分布特征:")
    print(f"- 出度变异系数: {out_degrees.std() / out_degrees.mean():.4f}")
    print(f"- 入度变异系数: {in_degrees.std() / in_degrees.mean():.4f}")
    
    # 幂律分布检验（简单版）
    # 计算度分布的对数斜率
    degree_dist = np.bincount(out_degrees)
    non_zero_degrees = np.where(degree_dist > 0)[0][1:]  # 排除度为0
    if len(non_zero_degrees) > 1:
        log_degrees = np.log(non_zero_degrees)
        log_counts = np.log(degree_dist[non_zero_degrees])
        slope = np.polyfit(log_degrees, log_counts, 1)[0]
        print(f"- 出度分布幂律指数估计: {abs(slope):.2f}")
        
        if abs(slope) > 1.5 and abs(slope) < 3.5:
            print("- 评估: 度分布接近幂律分布，符合真实网络特征")
        else:
            print("- 评估: 度分布可能不符合典型的幂律分布")
    
    # 稀疏性
    sparsity = num_edges / (num_nodes * (num_nodes - 1))
    print(f"\n网络稀疏性: {sparsity:.6f}")
    if sparsity < 0.001:
        print("- 评估: 网络非常稀疏，符合大规模真实网络特征")
    elif sparsity < 0.01:
        print("- 评估: 网络较稀疏，可能适合中等规模应用")
    else:
        print("- 评估: 网络较密集，可能不太真实")
    
    return sparsity

def evaluate_temporal_locality(df, dataset_name, sample_size=100000):
    """评估时序局部性"""
    print(f"\n### 时序局部性评估 ###")
    
    # 采样分析（对大数据集）
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    # 计算节点的时间跨度
    node_timespan = {}
    for _, row in df_sample.iterrows():
        src, tgt, ts = row['source'], row['target'], row['timestamp']
        
        if src not in node_timespan:
            node_timespan[src] = [ts, ts]
        else:
            node_timespan[src][0] = min(node_timespan[src][0], ts)
            node_timespan[src][1] = max(node_timespan[src][1], ts)
            
        if tgt not in node_timespan:
            node_timespan[tgt] = [ts, ts]
        else:
            node_timespan[tgt][0] = min(node_timespan[tgt][0], ts)
            node_timespan[tgt][1] = max(node_timespan[tgt][1], ts)
    
    # 计算活跃时间跨度
    timespans = [span[1] - span[0] for span in node_timespan.values() if span[1] > span[0]]
    if timespans:
        avg_timespan = np.mean(timespans)
        total_timespan = df['timestamp'].max() - df['timestamp'].min()
        
        print(f"节点活跃度分析:")
        print(f"- 平均节点活跃时间跨度: {avg_timespan:.0f} 秒")
        print(f"- 相对活跃度: {avg_timespan/total_timespan:.4f}")
        
        if avg_timespan/total_timespan < 0.5:
            print("- 评估: 节点具有明显的时序局部性，适合时序预测任务")
        else:
            print("- 评估: 节点活跃时间较长，时序特征可能不明显")

def evaluate_with_tgx(df, dataset_name):
    """使用 TGX 进行评估"""
    if not HAS_TGX:
        return
        
    print(f"\n### TGX 专业评估 ###")
    
    try:
        # 保存为 CSV 供 TGX 使用
        temp_csv = f"{dataset_name}_temp.csv"
        df.to_csv(temp_csv, index=False)
        
        # 创建 TGX 数据集
        dataset = tgx.tgx_utils.read_csv(temp_csv)
        
        # TGX 统计
        print("TGX 计算的统计信息:")
        print(f"- 节点数: {dataset.num_nodes()}")
        print(f"- 边数: {dataset.num_edges()}")
        print(f"- 平均度: {dataset.avg_degree():.2f}")
        
        # 计算更多 TGX 特征
        if hasattr(dataset, 'get_reoccurrence'):
            reoccur = dataset.get_reoccurrence()
            print(f"- 边重现率: {reoccur:.4f}")
            
        if hasattr(dataset, 'get_surprise'):
            surprise = dataset.get_surprise()
            print(f"- 惊奇度: {surprise:.4f}")
        
        # 清理临时文件
        os.remove(temp_csv)
        
    except Exception as e:
        print(f"TGX 评估时出错: {e}")

def generate_evaluation_report(df, dataset_name):
    """生成综合评估报告"""
    print(f"\n{'='*60}")
    print(f"数据集质量评估报告: {dataset_name}")
    print(f"{'='*60}")
    
    # 1. 基本信息
    print("\n### 基本信息 ###")
    all_nodes = pd.concat([df['source'], df['target']]).unique()
    print(f"- 节点数: {len(all_nodes):,}")
    print(f"- 边数: {len(df):,}")
    print(f"- 唯一时间戳数: {df['timestamp'].nunique():,}")
    
    # 2. 时序模式评估
    temporal_density = evaluate_temporal_patterns(df, dataset_name)
    
    # 3. 图结构评估
    sparsity = evaluate_graph_structure(df, dataset_name)
    
    # 4. 时序局部性评估
    evaluate_temporal_locality(df, dataset_name)
    
    # 5. TGX 评估
    evaluate_with_tgx(df, dataset_name)
    
    # 6. 总体评分
    print(f"\n### 总体评估 ###")
    score = 0
    reasons = []
    
    # 时序特征评分
    if temporal_density > 0.8:
        score += 30
        reasons.append("优秀的时序稀疏性 (+30)")
    elif temporal_density > 0.5:
        score += 20
        reasons.append("良好的时序稀疏性 (+20)")
    else:
        score += 10
        reasons.append("时序分布较密集 (+10)")
    
    # 网络稀疏性评分
    if sparsity < 0.001:
        score += 30
        reasons.append("优秀的网络稀疏性 (+30)")
    elif sparsity < 0.01:
        score += 20
        reasons.append("良好的网络稀疏性 (+20)")
    else:
        score += 10
        reasons.append("网络较密集 (+10)")
    
    # 规模评分
    if len(all_nodes) > 500000:
        score += 20
        reasons.append("大规模网络 (+20)")
    elif len(all_nodes) > 100000:
        score += 15
        reasons.append("中等规模网络 (+15)")
    else:
        score += 10
        reasons.append("小规模网络 (+10)")
    
    # 边数评分
    if len(df) > 5000000:
        score += 20
        reasons.append("海量边数据 (+20)")
    elif len(df) > 1000000:
        score += 15
        reasons.append("大量边数据 (+15)")
    else:
        score += 10
        reasons.append("中等边数据 (+10)")
    
    print(f"\n综合评分: {score}/100")
    print("评分细节:")
    for reason in reasons:
        print(f"  - {reason}")
    
    print(f"\n### 使用建议 ###")
    if score >= 80:
        print("✓ 非常适合作为时序图生成任务的基准数据集")
        print("✓ 具有真实世界网络的典型特征")
        print("✓ 可用于评估时序图生成模型的质量")
    elif score >= 60:
        print("✓ 可以作为时序图生成任务的数据集")
        print("! 某些特征可能需要改进")
        print("! 建议与其他数据集对比使用")
    else:
        print("! 作为生成任务基准可能存在局限")
        print("! 建议进一步预处理或选择其他数据集")

def main():
    """主函数"""
    dataset_path = os.path.expanduser("~/dataset/")
    
    datasets = {
        'citation_network_10M': 'citation_network_10M.txt',
        'social_media_100M': 'social_media_100M.txt'
    }
    
    # 保存报告
    report_file = open('evaluation_report.txt', 'w', encoding='utf-8')
    report_file.write(f"时序图数据集评估报告\n")
    report_file.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_file.write("="*60 + "\n\n")
    
    for name, filename in datasets.items():
        file_path = os.path.join(dataset_path, filename)
        
        # 对于100M的数据集，只分析一部分
        if '100M' in name:
            df = load_dataset(file_path, name, sample_size=10000000)
        else:
            df = load_dataset(file_path, name)
        
        if df is not None:
            # 生成评估报告
            generate_evaluation_report(df, name)
            
            # 清理内存
            del df
            print(f"\n{name} 评估完成！")
            print("\n" + "="*80 + "\n")
    
    report_file.close()
    print("\n所有评估完成！详细报告已保存到 evaluation_report.txt")

if __name__ == "__main__":
    main()