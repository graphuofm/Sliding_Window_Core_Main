#!/usr/bin/env python3
"""
Dataset Analysis and TGX Evaluation Script
Generates comprehensive statistics for temporal graph datasets
"""

import os
import sys
import time
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

# Try to import TGX
try:
    import tgx
    from tgx.io.read import read_csv
    TGX_AVAILABLE = True
    print("✅ TGX imported successfully")
except ImportError:
    TGX_AVAILABLE = False
    print("❌ TGX not available. Install with: pip install py-tgx")

def analyze_basic_stats(filepath):
    """Analyze basic statistics of a temporal graph file"""
    print(f"\n📊 Analyzing {os.path.basename(filepath)}...")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    # Get file size
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    nodes = set()
    edges = 0
    timestamps = []
    degrees = defaultdict(int)
    
    start_time = time.time()
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if len(parts) >= 3:
                try:
                    u, v, t = int(parts[0]), int(parts[1]), int(parts[2])
                    nodes.add(u)
                    nodes.add(v)
                    edges += 1
                    timestamps.append(t)
                    degrees[u] += 1
                    degrees[v] += 1
                except ValueError:
                    continue
            
            # Progress indicator for large files
            if line_num % 1000000 == 0:
                print(f"  Processed {line_num:,} lines...")
    
    processing_time = time.time() - start_time
    
    if not timestamps:
        print(f"❌ No valid edges found in {filepath}")
        return None
    
    # Calculate statistics
    timestamps = np.array(timestamps)
    degree_values = list(degrees.values())
    
    stats = {
        'filename': os.path.basename(filepath),
        'file_size_mb': round(file_size_mb, 1),
        'num_nodes': len(nodes),
        'num_edges': edges,
        'min_timestamp': int(timestamps.min()),
        'max_timestamp': int(timestamps.max()),
        'time_span_days': int((timestamps.max() - timestamps.min()) / (24 * 3600)),
        'time_span_years': round((timestamps.max() - timestamps.min()) / (365 * 24 * 3600), 2),
        'max_degree': max(degree_values) if degree_values else 0,
        'avg_degree': round(np.mean(degree_values), 2) if degree_values else 0,
        'density': round(2 * edges / (len(nodes) * (len(nodes) - 1)), 8) if len(nodes) > 1 else 0,
        'processing_time_sec': round(processing_time, 1)
    }
    
    return stats

def analyze_with_tgx(filepath, dataset_name):
    """Analyze dataset using TGX library"""
    if not TGX_AVAILABLE:
        print(f"⚠️  Skipping TGX analysis for {dataset_name} (TGX not available)")
        return None
    
    print(f"\n🔬 TGX Analysis for {dataset_name}...")
    
    try:
        # Load data with TGX
        print("  Loading data...")
        edgelist = read_csv(filepath, header=False, t_col=2, src_col=0, dst_col=1)
        ctdg = tgx.Graph(edgelist=edgelist)
        
        print("  Computing basic TGX statistics...")
        tgx_stats = {
            'tgx_nodes': ctdg.number_of_nodes(),
            'tgx_edges': ctdg.number_of_edges(),
            'tgx_time_range': ctdg.get_time_range()
        }
        
        # Discretize for temporal analysis (use appropriate time scale)
        print("  Discretizing temporal graph...")
        if tgx_stats['tgx_edges'] > 10000000:  # Large datasets
            time_scale = "monthly"
        elif tgx_stats['tgx_edges'] > 1000000:  # Medium datasets  
            time_scale = "weekly"
        else:  # Small datasets
            time_scale = "daily"
            
        dtdg, ts_list = ctdg.discretize(time_scale=time_scale, store_unix=True)
        
        print(f"  Computing temporal metrics (time_scale={time_scale})...")
        
        # Compute TGX-specific metrics
        try:
            novelty = tgx.get_novelty(dtdg)
            tgx_stats['novelty_index'] = round(novelty, 4)
        except:
            tgx_stats['novelty_index'] = "N/A"
            
        try:
            surprise = tgx.get_surprise(dtdg)
            tgx_stats['surprise_index'] = round(surprise, 4)
        except:
            tgx_stats['surprise_index'] = "N/A"
            
        # Activity patterns
        try:
            # Get number of active time windows
            active_windows = len([t for t in ts_list if dtdg.number_of_edges_at_timestamp(t) > 0])
            tgx_stats['active_time_windows'] = active_windows
            tgx_stats['temporal_density'] = round(active_windows / len(ts_list), 4)
        except:
            tgx_stats['active_time_windows'] = "N/A"
            tgx_stats['temporal_density'] = "N/A"
        
        return tgx_stats
        
    except Exception as e:
        print(f"❌ TGX analysis failed for {dataset_name}: {str(e)}")
        return None

def generate_latex_table(all_stats):
    """Generate LaTeX table for paper"""
    print("\n📋 Generating LaTeX table...")
    
    latex = """\\begin{table}[htbp]
\\centering
\\caption{Dataset Characteristics}
\\label{tab:datasets}
\\begin{tabular}{l|c|c|c|c|c|c}
\\hline
\\textbf{Dataset} & \\textbf{Nodes} & \\textbf{Edges} & \\textbf{Time Span} & \\textbf{Max Degree} & \\textbf{Density} & \\textbf{Type} \\\\
\\hline
"""
    
    # Define dataset types
    dataset_types = {
        'bitcoin-temporal.txt': 'Transaction',
        'cit-Patents.txt': 'Citation', 
        'soc-LiveJournal1.txt': 'Social',
        'sx-stackoverflow.txt': 'Q\\&A',
        'sx-superuser.txt': 'Q\\&A',
        'temporal-reddit-reply.txt': 'Social',
        'wiki-talk-temporal.txt': 'Communication',
        'citation_network_10M.txt': 'Synthetic (Dense)',
        'social_media_100M.txt': 'Synthetic (Dense)', 
        'communication_1B.txt': 'Synthetic (Uniform)'
    }
    
    for stats in all_stats:
        if stats is None:
            continue
            
        filename = stats['filename']
        dataset_type = dataset_types.get(filename, 'Unknown')
        
        # Format time span
        if stats['time_span_years'] >= 1:
            time_span = f"{stats['time_span_years']:.1f}y"
        else:
            time_span = f"{stats['time_span_days']}d"
            
        # Format numbers
        nodes = f"{stats['num_nodes']:,}"
        edges = f"{stats['num_edges']:,}"
        density = f"{stats['density']:.2e}"
        
        # Format filename for LaTeX (escape underscores)
        latex_filename = filename.replace('_', '\\_').replace('.txt', '')
        latex += f"{latex_filename} & {nodes} & {edges} & {time_span} & {stats['max_degree']} & {density} & {dataset_type} \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\end{table}
"""
    
    return latex

def main():
    print("🚀 Dataset Analysis and TGX Evaluation")
    print("=" * 60)
    
    # Dataset directory
    dataset_dir = "dataset"
    
    # List of datasets to analyze
    datasets = [
        'bitcoin-temporal.txt',
        'cit-Patents.txt', 
        'soc-LiveJournal1.txt',
        'sx-stackoverflow.txt',
        'sx-superuser.txt', 
        'temporal-reddit-reply.txt',
        'wiki-talk-temporal.txt',
        'citation_network_10M.txt',
        'social_media_100M.txt',
        'communication_1B.txt'
    ]
    
    all_stats = []
    tgx_results = {}
    
    # Analyze each dataset
    for dataset in datasets:
        filepath = os.path.join(dataset_dir, dataset)
        
        # Basic statistics
        stats = analyze_basic_stats(filepath)
        if stats:
            all_stats.append(stats)
            
            # TGX analysis for synthetic datasets
            if 'synthetic' in dataset.lower() or dataset in ['citation_network_10M.txt', 'social_media_100M.txt', 'communication_1B.txt']:
                tgx_stats = analyze_with_tgx(filepath, dataset)
                if tgx_stats:
                    tgx_results[dataset] = tgx_stats
    
    # Print summary statistics
    print("\n" + "="*80)
    print("📊 DATASET SUMMARY STATISTICS")
    print("="*80)
    
    for stats in all_stats:
        print(f"\n📁 {stats['filename']}")
        print(f"   Nodes: {stats['num_nodes']:,}")
        print(f"   Edges: {stats['num_edges']:,}")
        print(f"   Time Span: {stats['time_span_years']:.2f} years ({stats['time_span_days']} days)")
        print(f"   Max Degree: {stats['max_degree']}")
        print(f"   Avg Degree: {stats['avg_degree']}")
        print(f"   Density: {stats['density']:.2e}")
        print(f"   File Size: {stats['file_size_mb']} MB")
    
    # Print TGX results for synthetic datasets
    if tgx_results:
        print("\n" + "="*80) 
        print("🔬 TGX EVALUATION RESULTS (Synthetic Datasets)")
        print("="*80)
        
        for dataset, tgx_stats in tgx_results.items():
            print(f"\n🎯 {dataset}")
            for key, value in tgx_stats.items():
                print(f"   {key}: {value}")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(all_stats)
    
    print("\n" + "="*80)
    print("📋 LATEX TABLE FOR PAPER")
    print("="*80)
    print(latex_table)
    
    # Save results to files
    print("\n💾 Saving results...")
    
    # Save statistics to CSV
    df_stats = pd.DataFrame(all_stats)
    df_stats.to_csv('dataset_statistics.csv', index=False)
    print("✅ Saved dataset_statistics.csv")
    
    # Save LaTeX table
    with open('dataset_table.tex', 'w') as f:
        f.write(latex_table)
    print("✅ Saved dataset_table.tex")
    
    # Save TGX results
    if tgx_results:
        df_tgx = pd.DataFrame(tgx_results).T
        df_tgx.to_csv('tgx_evaluation.csv')
        print("✅ Saved tgx_evaluation.csv")
    
    print("\n🎉 Analysis complete! Ready for paper writing!")

if __name__ == "__main__":
    main()
    