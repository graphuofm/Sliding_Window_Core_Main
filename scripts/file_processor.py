import os
import random

def process_patents_file(file_path):
    """
    处理cit-Patents.txt文件，添加时间戳
    格式：node1 node2 -> node1 node2 timestamp
    """
    print(f"Processing {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return
    
    # 读取原文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Original file has {len(lines)} lines")
    
    # 生成时间戳范围（假设为5年的专利引用数据）
    start_timestamp = 1000000000  # 2001年
    end_timestamp = 1157760000    # 2006年
    
    processed_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:  # 跳过空行
            continue
            
        # 假设原格式是 "node1 node2" 或 "node1\tnode2"
        parts = line.split()
        if len(parts) >= 2:
            node1, node2 = parts[0], parts[1]
            # 生成随机时间戳
            timestamp = random.randint(start_timestamp, end_timestamp)
            processed_lines.append(f"{node1} {node2} {timestamp}\n")
        
        # 显示进度
        if (i + 1) % 10000 == 0:  # 更频繁的进度显示
            print(f"Processed {i + 1} lines...")
    
    # 按时间戳排序
    print("Sorting by timestamp...")
    processed_lines.sort(key=lambda x: int(x.split()[2]))
    
    # 写回原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)
    
    print(f"✓ Processed {len(processed_lines)} edges and saved to {file_path}")
    return processed_lines

def process_livejournal_file(file_path):
    """
    处理soc-LiveJournal.txt文件，删除#开头的注释行，添加时间戳
    """
    print(f"Processing {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return
    
    # 读取原文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Original file has {len(lines)} lines")
    
    # 删除注释行
    data_lines = []
    comment_count = 0
    
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            comment_count += 1
            continue
        if not line:  # 跳过空行
            continue
        data_lines.append(line)
    
    print(f"Removed {comment_count} comment lines")
    print(f"Remaining {len(data_lines)} data lines")
    
    # 生成时间戳范围（假设为社交网络3年数据）
    start_timestamp = 1200000000  # 2008年
    end_timestamp = 1300000000    # 2011年
    
    processed_lines = []
    
    for i, line in enumerate(data_lines):
        parts = line.split()
        if len(parts) >= 2:
            node1, node2 = parts[0], parts[1]
            # 生成随机时间戳
            timestamp = random.randint(start_timestamp, end_timestamp)
            processed_lines.append(f"{node1} {node2} {timestamp}\n")
        
        # 显示进度
        if (i + 1) % 10000 == 0:  # 更频繁的进度显示
            print(f"Processed {i + 1} lines...")
    
    # 按时间戳排序
    print("Sorting by timestamp...")
    processed_lines.sort(key=lambda x: int(x.split()[2]))
    
    # 写回原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(processed_lines)
    
    print(f"✓ Processed {len(processed_lines)} edges and saved to {file_path}")
    return processed_lines

def print_first_lines(file_path, num_lines=20):
    """打印文件的前几行"""
    print(f"\n--- First {num_lines} lines of {os.path.basename(file_path)} ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i in range(num_lines):
                line = f.readline()
                if not line:
                    break
                print(f"{i+1:2d}: {line.rstrip()}")
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    print("Graph File Processor (Server Version)")
    print("=" * 50)
    print("Starting file processing...")
    
    # 服务器上的文件路径（相对路径）
    patents_file = "cit-Patents.txt"
    livejournal_file = "soc-LiveJournal.txt"
    
    print(f"Patents file: {patents_file}")
    print(f"LiveJournal file: {livejournal_file}")
    
    # 检查文件是否存在
    print(f"Patents file exists: {os.path.exists(patents_file)}")
    print(f"LiveJournal file exists: {os.path.exists(livejournal_file)}")
    
    # 设置随机种子以便复现
    random.seed(42)
    print("Random seed set to 42")
    
    # 处理Patents文件
    print("\n1. Processing Patents citation network...")
    try:
        process_patents_file(patents_file)
        print_first_lines(patents_file)
    except Exception as e:
        print(f"Error processing Patents file: {e}")
    
    print("\n" + "="*70)
    
    # 处理LiveJournal文件
    print("\n2. Processing LiveJournal social network...")
    try:
        process_livejournal_file(livejournal_file)
        print_first_lines(livejournal_file)
    except Exception as e:
        print(f"Error processing LiveJournal file: {e}")
    
    print("\n✅ All files processed successfully!")
    print("\nBoth files now have the format: node1 node2 timestamp")

if __name__ == "__main__":
    main()