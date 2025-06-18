#!/usr/bin/env python3
"""
实验1: NetworkX k-core计算
用于验证正确性和性能比较的基准
"""

import networkx as nx
import time
import sys
import os

def load_temporal_graph(filepath):
    """加载时序图数据，忽略时间戳"""
    G = nx.Graph()
    edge_count = 0
    self_loop_count = 0
    
    print(f"Loading graph from {filepath}...")
    start_time = time.time()
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%') or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                
                # 跳过自环边
                if u == v:
                    self_loop_count += 1
                    continue
                    
                G.add_edge(u, v)
                edge_count += 1
                
                if edge_count % 100000 == 0:
                    print(f"  Loaded {edge_count} edges...")
    
    load_time = time.time() - start_time
    print(f"Graph loaded in {load_time:.3f} seconds")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Self-loops skipped: {self_loop_count}")
    
    return G

def compute_kcore_networkx(G):
    """使用NetworkX计算k-core分解"""
    print("\nComputing k-core decomposition with NetworkX...")
    start_time = time.time()
    
    # 确保没有自环边
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # NetworkX的core_number函数返回每个节点的核心数
    core_numbers = nx.core_number(G)
    
    compute_time = time.time() - start_time
    print(f"K-core computation completed in {compute_time:.3f} seconds")
    
    return core_numbers, compute_time

def save_results(core_numbers, output_file):
    """保存核心数结果到文件"""
    print(f"\nSaving results to {output_file}...")
    
    # 转换为排序的列表以便比较
    sorted_cores = sorted(core_numbers.items())
    
    with open(output_file, 'w') as f:
        f.write(f"# NetworkX k-core results\n")
        f.write(f"# Format: vertex_id core_number\n")
        f.write(f"# Total vertices: {len(core_numbers)}\n")
        for vertex, core in sorted_cores:
            f.write(f"{vertex} {core}\n")
    
    print(f"Results saved. Total vertices: {len(core_numbers)}")

def analyze_core_distribution(core_numbers):
    """分析核心数分布"""
    print("\nCore number distribution:")
    
    # 统计每个核心数的顶点数量
    core_dist = {}
    max_core = 0
    
    for vertex, core in core_numbers.items():
        if core not in core_dist:
            core_dist[core] = 0
        core_dist[core] += 1
        max_core = max(max_core, core)
    
    print(f"  Max core number: {max_core}")
    print(f"  Core distribution (showing first 10 and last 5):")
    
    sorted_cores = sorted(core_dist.items())
    
    # 显示前10个
    for i, (core, count) in enumerate(sorted_cores[:10]):
        print(f"    {core}-core: {count} vertices")
    
    if len(sorted_cores) > 15:
        print("    ...")
        # 显示最后5个
        for core, count in sorted_cores[-5:]:
            print(f"    {core}-core: {count} vertices")
    elif len(sorted_cores) > 10:
        # 如果总数在10-15之间，显示剩余的
        for i in range(10, len(sorted_cores)):
            core, count = sorted_cores[i]
            print(f"    {core}-core: {count} vertices")

def main():
    # 设置数据集路径
    dataset_path = "/home/jding/dataset/sx-superuser.txt"
    output_file = "/home/jding/e1_networkx_results.txt"
    
    print("=" * 60)
    print("Experiment 1: NetworkX K-core Computation")
    print("=" * 60)
    
    # 加载图
    G = load_temporal_graph(dataset_path)
    
    # 计算k-core
    core_numbers, compute_time = compute_kcore_networkx(G)
    
    # 分析结果
    analyze_core_distribution(core_numbers)
    
    # 保存结果
    save_results(core_numbers, output_file)
    
    # 性能总结
    print("\n" + "=" * 60)
    print("Performance Summary:")
    print(f"  Total computation time: {compute_time:.3f} seconds")
    print(f"  Vertices processed: {len(core_numbers)}")
    print(f"  Edges processed: {G.number_of_edges()}")
    print(f"  Throughput: {G.number_of_edges() / compute_time:.0f} edges/second")
    print("=" * 60)

if __name__ == "__main__":
    main()