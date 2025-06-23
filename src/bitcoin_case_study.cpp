#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <climits>
#include <map>
#include <ctime>
#include <omp.h>

using namespace std;
using vertex_t = int;
using edge_t = pair<vertex_t, vertex_t>;
using timestamp_t = long long;

// 时序边结构
struct TemporalEdge {
    vertex_t src;
    vertex_t dst;
    timestamp_t timestamp;
    
    TemporalEdge(vertex_t s, vertex_t d, timestamp_t t) : src(s), dst(d), timestamp(t) {}
};

// 核心度统计信息
struct CoreStats {
    int day;
    int max_core;
    int num_vertices_in_max_core;
    double processing_time_ms;
    timestamp_t date_timestamp;
    map<int, int> core_distribution; // core_value -> count
};

// 高精度计时器
class Timer {
private:
    chrono::high_resolution_clock::time_point start_time;
public:
    Timer() : start_time(chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_time = chrono::high_resolution_clock::now();
    }
    
    double elapsed_milliseconds() const {
        auto end_time = chrono::high_resolution_clock::now();
        return chrono::duration<double, milli>(end_time - start_time).count();
    }
    
    double elapsed_seconds() const {
        return elapsed_milliseconds() / 1000.0;
    }
};

// 图结构
class Graph {
public:
    vector<vector<vertex_t>> adj;
    vector<int> core;
    size_t num_vertices;
    
    Graph() : num_vertices(0) {}
    
    void ensure_vertex(vertex_t v) {
        if (v >= num_vertices) {
            num_vertices = v + 1;
            adj.resize(num_vertices);
            core.resize(num_vertices, 0);
        }
    }
    
    void add_edge(vertex_t u, vertex_t v) {
        if (u == v) return;
        
        ensure_vertex(max(u, v));
        
        if (find(adj[u].begin(), adj[u].end(), v) == adj[u].end()) {
            adj[u].push_back(v);
        }
        if (find(adj[v].begin(), adj[v].end(), u) == adj[v].end()) {
            adj[v].push_back(u);
        }
    }
    
    void remove_edge(vertex_t u, vertex_t v) {
        if (u >= num_vertices || v >= num_vertices) return;
        
        auto it_u = find(adj[u].begin(), adj[u].end(), v);
        if (it_u != adj[u].end()) {
            *it_u = adj[u].back();
            adj[u].pop_back();
        }
        
        auto it_v = find(adj[v].begin(), adj[v].end(), u);
        if (it_v != adj[v].end()) {
            *it_v = adj[v].back();
            adj[v].pop_back();
        }
    }
    
    void compute_core_numbers_bz() {
        if (num_vertices == 0) return;
        
        fill(core.begin(), core.end(), 0);
        
        vector<int> degree(num_vertices);
        int max_degree = 0;
        for (vertex_t v = 0; v < num_vertices; ++v) {
            degree[v] = adj[v].size();
            max_degree = max(max_degree, degree[v]);
        }
        
        if (max_degree == 0) return;
        
        vector<vector<vertex_t>> bins(max_degree + 1);
        for (vertex_t v = 0; v < num_vertices; ++v) {
            bins[degree[v]].push_back(v);
        }
        
        vector<bool> processed(num_vertices, false);
        
        for (int d = 0; d <= max_degree; ++d) {
            for (vertex_t v : bins[d]) {
                if (processed[v]) continue;
                
                core[v] = d;
                processed[v] = true;
                
                for (vertex_t w : adj[v]) {
                    if (processed[w]) continue;
                    
                    if (degree[w] > d) {
                        auto& bin = bins[degree[w]];
                        auto it = find(bin.begin(), bin.end(), w);
                        if (it != bin.end()) {
                            swap(*it, bin.back());
                            bin.pop_back();
                        }
                        
                        --degree[w];
                        bins[degree[w]].push_back(w);
                    }
                }
            }
        }
    }
    
    Graph copy() const {
        Graph g;
        g.num_vertices = num_vertices;
        g.adj = adj;
        g.core = core;
        return g;
    }
    
    void clear() {
        adj.clear();
        core.clear();
        num_vertices = 0;
    }
    
    // 获取核心度统计信息
    CoreStats get_core_stats() const {
        CoreStats stats;
        stats.max_core = 0;
        stats.num_vertices_in_max_core = 0;
        
        for (vertex_t v = 0; v < num_vertices; ++v) {
            if (core[v] > 0) {
                stats.core_distribution[core[v]]++;
                if (core[v] > stats.max_core) {
                    stats.max_core = core[v];
                    stats.num_vertices_in_max_core = 1;
                } else if (core[v] == stats.max_core) {
                    stats.num_vertices_in_max_core++;
                }
            }
        }
        
        return stats;
    }
};

// UCR批处理版本（优化版）
class UCRBatch {
private:
    Graph& G;
    vector<int> r_values;
    vector<int> s_values;
    
    bool is_qualified(vertex_t v) const {
        if (v >= r_values.size()) return false;
        int k = G.core[v];
        return r_values[v] + s_values[v] > k;
    }
    
    void update_s_value(vertex_t v) {
        if (v >= G.num_vertices) return;
        int k = G.core[v];
        int s_count = 0;
        
        for (vertex_t w : G.adj[v]) {
            if (G.core[w] == k && is_qualified(w)) {
                ++s_count;
            }
        }
        
        if (v < s_values.size()) {
            s_values[v] = s_count;
        }
    }
    
public:
    UCRBatch(Graph& g) : G(g) {
        reset();
    }
    
    void reset() {
        size_t n = G.num_vertices;
        r_values.resize(n, 0);
        s_values.resize(n, 0);
        
        // 计算r值
        for (vertex_t v = 0; v < n; ++v) {
            int k = G.core[v];
            int r_count = 0;
            
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] > k) {
                    ++r_count;
                }
            }
            
            r_values[v] = r_count;
        }
        
        // 计算s值
        for (vertex_t v = 0; v < n; ++v) {
            update_s_value(v);
        }
    }
    
    double process_sliding_window(const vector<edge_t>& remove_edges, 
                                 const vector<edge_t>& add_edges) {
        Timer timer;
        
        // 扩展数据结构
        vertex_t max_vertex_id = 0;
        for (const auto& edge : add_edges) {
            max_vertex_id = max(max_vertex_id, max(edge.first, edge.second));
        }
        
        if (max_vertex_id >= G.core.size()) {
            G.core.resize(max_vertex_id + 1, 0);
            r_values.resize(max_vertex_id + 1, 0);
            s_values.resize(max_vertex_id + 1, 0);
        }
        
        // Phase 1: 批量删除边
        unordered_map<int, unordered_set<vertex_t>> k_groups_remove;
        
        for (const auto& edge : remove_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u < G.num_vertices && v < G.num_vertices) {
                int ku = G.core[u];
                int kv = G.core[v];
                
                if (ku < kv) r_values[u]--;
                else if (ku > kv) r_values[v]--;
                
                if (ku > 0) k_groups_remove[ku].insert(u);
                if (kv > 0) k_groups_remove[kv].insert(v);
                
                G.remove_edge(u, v);
            }
        }
        
        // 批量处理核心度降级
        vector<int> k_values;
        for (const auto& pair : k_groups_remove) {
            k_values.push_back(pair.first);
        }
        sort(k_values.begin(), k_values.end());
        
        for (int k : k_values) {
            for (vertex_t v : k_groups_remove[k]) {
                if (G.core[v] == k && r_values[v] + s_values[v] < k) {
                    G.core[v] = k - 1;
                }
            }
        }
        
        // Phase 2: 批量添加边
        unordered_set<vertex_t> affected_vertices;
        
        for (const auto& edge : add_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u == v) continue;
            
            G.ensure_vertex(max(u, v));
            
            G.add_edge(u, v);
            
            int ku = G.core[u];
            int kv = G.core[v];
            
            if (ku < kv) r_values[u]++;
            else if (ku > kv) r_values[v]++;
            
            affected_vertices.insert(u);
            affected_vertices.insert(v);
        }
        
        // 批量更新s值
        for (vertex_t v : affected_vertices) {
            update_s_value(v);
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] == G.core[v]) {
                    update_s_value(w);
                }
            }
        }
        
        // 批量处理核心度升级
        for (vertex_t v : affected_vertices) {
            if (r_values[v] + s_values[v] > G.core[v]) {
                G.core[v]++;
            }
        }
        
        return timer.elapsed_milliseconds();
    }
};

// 时间戳转日期字符串
string timestamp_to_date(timestamp_t ts) {
    time_t time = static_cast<time_t>(ts);
    char buffer[100];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d", gmtime(&time));
    return string(buffer);
}

// 检测异常
bool detect_anomaly(const CoreStats& current, const CoreStats& previous) {
    // 检测最大核心度的剧烈变化
    if (previous.max_core > 0) {
        double change_ratio = abs(current.max_core - previous.max_core) / (double)previous.max_core;
        if (change_ratio > 0.3) { // 30%以上的变化
            return true;
        }
    }
    
    // 检测大核心的突然消失
    if (previous.max_core > 50 && current.max_core < 30) {
        return true;
    }
    
    return false;
}

int main() {
    cout << "===========================================\n";
    cout << "Bitcoin Transaction Network Case Study\n";
    cout << "Monitoring with Sliding Window k-Core\n";
    cout << "===========================================\n\n";
    
    // 参数设置
    const int window_days = 7;  // 7天窗口
    const int slide_days = 1;   // 每天滑动
    const int seconds_per_day = 86400;
    
    // 读取Bitcoin数据集
    string filepath = "dataset/bitcoin-temporal.txt";
    ifstream file(filepath);
    if (!file.is_open()) {
        cout << "ERROR: Cannot open " << filepath << "\n";
        return 1;
    }
    
    cout << "Loading Bitcoin transaction data...\n";
    vector<TemporalEdge> all_edges;
    string line;
    timestamp_t min_ts = LLONG_MAX;
    timestamp_t max_ts = LLONG_MIN;
    
    Timer load_timer;
    size_t line_count = 0;
    
    while (getline(file, line)) {
        if (line.empty() || line[0] == '%' || line[0] == '#') continue;
        
        istringstream iss(line);
        vertex_t src, dst;
        timestamp_t ts;
        
        if (iss >> src >> dst >> ts && src != dst) {
            all_edges.emplace_back(src, dst, ts);
            min_ts = min(min_ts, ts);
            max_ts = max(max_ts, ts);
            
            line_count++;
            if (line_count % 1000000 == 0) {
                cout << "  Loaded " << line_count / 1000000 << "M edges...\n";
            }
        }
    }
    file.close();
    
    cout << "Loading complete: " << all_edges.size() << " edges in " 
         << load_timer.elapsed_seconds() << " seconds\n";
    cout << "Time span: " << timestamp_to_date(min_ts) << " to " << timestamp_to_date(max_ts) << "\n";
    
    // 按天分组边
    cout << "\nGrouping edges by day...\n";
    int total_days = (max_ts - min_ts) / seconds_per_day + 1;
    vector<vector<edge_t>> daily_edges(total_days);
    
    for (const auto& e : all_edges) {
        int day = (e.timestamp - min_ts) / seconds_per_day;
        daily_edges[day].push_back(make_pair(e.src, e.dst));
    }
    
    // 打开输出文件
    ofstream timeline_file("bitcoin_timeline.txt");
    ofstream anomaly_file("bitcoin_anomalies.txt");
    
    timeline_file << "Date\tDay\tMaxCore\tVerticesInMaxCore\tProcessingTime(ms)\n";
    anomaly_file << "Date\tDay\tEvent\tPrevMaxCore\tNewMaxCore\tChangeRatio\n";
    
    // 构建初始窗口
    cout << "\nBuilding initial window...\n";
    Graph G;
    for (int day = 0; day < window_days && day < total_days; ++day) {
        for (const auto& edge : daily_edges[day]) {
            G.add_edge(edge.first, edge.second);
        }
    }
    
    cout << "Computing initial core numbers...\n";
    Timer init_timer;
    G.compute_core_numbers_bz();
    cout << "Initial BZ computation: " << init_timer.elapsed_milliseconds() << " ms\n";
    
    // 初始化UCR
    UCRBatch ucr(G);
    
    // 滑动窗口监控
    cout << "\nStarting sliding window monitoring...\n";
    vector<CoreStats> timeline;
    CoreStats prev_stats = G.get_core_stats();
    
    for (int current_day = window_days; current_day < total_days - 1; current_day += slide_days) {
        if (current_day % 30 == 0) {
            cout << "Processing day " << current_day << "/" << total_days 
                 << " (" << timestamp_to_date(min_ts + current_day * seconds_per_day) << ")\n";
        }
        
        // 获取要删除和添加的边
        vector<edge_t> remove_edges = daily_edges[current_day - window_days];
        vector<edge_t> add_edges = daily_edges[current_day];
        
        // UCR批处理更新
        double processing_time = ucr.process_sliding_window(remove_edges, add_edges);
        
        // 获取统计信息
        CoreStats stats = G.get_core_stats();
        stats.day = current_day;
        stats.processing_time_ms = processing_time;
        stats.date_timestamp = min_ts + current_day * seconds_per_day;
        
        // 记录到时间线
        timeline.push_back(stats);
        timeline_file << timestamp_to_date(stats.date_timestamp) << "\t"
                     << stats.day << "\t"
                     << stats.max_core << "\t"
                     << stats.num_vertices_in_max_core << "\t"
                     << fixed << setprecision(2) << stats.processing_time_ms << "\n";
        timeline_file.flush();
        
        // 检测异常
        if (detect_anomaly(stats, prev_stats)) {
            double change_ratio = (prev_stats.max_core > 0) ? 
                abs(stats.max_core - prev_stats.max_core) / (double)prev_stats.max_core : 0;
            
            string event = (stats.max_core < prev_stats.max_core) ? "CRASH" : "SURGE";
            
            cout << "*** ANOMALY DETECTED on " << timestamp_to_date(stats.date_timestamp) 
                 << ": " << event << " - Max core changed from " << prev_stats.max_core 
                 << " to " << stats.max_core << " (" << fixed << setprecision(1) 
                 << change_ratio * 100 << "% change) ***\n";
            
            anomaly_file << timestamp_to_date(stats.date_timestamp) << "\t"
                        << stats.day << "\t"
                        << event << "\t"
                        << prev_stats.max_core << "\t"
                        << stats.max_core << "\t"
                        << fixed << setprecision(3) << change_ratio << "\n";
            anomaly_file.flush();
        }
        
        prev_stats = stats;
    }
    
    // 计算性能统计
    double total_processing_time = 0;
    for (const auto& stats : timeline) {
        total_processing_time += stats.processing_time_ms;
    }
    
    double avg_processing_time = total_processing_time / timeline.size();
    
    cout << "\n===========================================\n";
    cout << "Case Study Summary:\n";
    cout << "Total days processed: " << timeline.size() << "\n";
    cout << "Average UCR processing time: " << fixed << setprecision(2) 
         << avg_processing_time << " ms per window update\n";
    cout << "Total processing time: " << fixed << setprecision(2) 
         << total_processing_time / 1000 << " seconds\n";
    cout << "Anomalies detected: Check bitcoin_anomalies.txt\n";
    cout << "Full timeline: Check bitcoin_timeline.txt\n";
    cout << "===========================================\n";
    
    // 输出可视化脚本
    ofstream viz_script("visualize_bitcoin.py");
    viz_script << "import pandas as pd\n";
    viz_script << "import matplotlib.pyplot as plt\n";
    viz_script << "import matplotlib.dates as mdates\n\n";
    viz_script << "# Read timeline data\n";
    viz_script << "df = pd.read_csv('bitcoin_timeline.txt', sep='\\t')\n";
    viz_script << "df['Date'] = pd.to_datetime(df['Date'])\n\n";
    viz_script << "# Read anomaly data\n";
    viz_script << "anomalies = pd.read_csv('bitcoin_anomalies.txt', sep='\\t')\n";
    viz_script << "anomalies['Date'] = pd.to_datetime(anomalies['Date'])\n\n";
    viz_script << "# Plot\n";
    viz_script << "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)\n\n";
    viz_script << "# Max core over time\n";
    viz_script << "ax1.plot(df['Date'], df['MaxCore'], 'b-', linewidth=1)\n";
    viz_script << "ax1.set_ylabel('Maximum k-Core', fontsize=12)\n";
    viz_script << "ax1.set_title('Bitcoin Transaction Network Core Evolution (2009-2016)', fontsize=14)\n";
    viz_script << "ax1.grid(True, alpha=0.3)\n\n";
    viz_script << "# Mark anomalies\n";
    viz_script << "for _, anomaly in anomalies.iterrows():\n";
    viz_script << "    ax1.axvline(x=anomaly['Date'], color='red', alpha=0.5, linestyle='--')\n";
    viz_script << "    ax1.text(anomaly['Date'], ax1.get_ylim()[1]*0.9, anomaly['Event'], \n";
    viz_script << "             rotation=90, verticalalignment='top', fontsize=8)\n\n";
    viz_script << "# Processing time\n";
    viz_script << "ax2.plot(df['Date'], df['ProcessingTime(ms)'], 'g-', linewidth=1)\n";
    viz_script << "ax2.set_ylabel('UCR Processing Time (ms)', fontsize=12)\n";
    viz_script << "ax2.set_xlabel('Date', fontsize=12)\n";
    viz_script << "ax2.grid(True, alpha=0.3)\n\n";
    viz_script << "# Format x-axis\n";
    viz_script << "ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))\n";
    viz_script << "ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))\n";
    viz_script << "plt.xticks(rotation=45)\n\n";
    viz_script << "plt.tight_layout()\n";
    viz_script << "plt.savefig('bitcoin_core_evolution.png', dpi=300)\n";
    viz_script << "plt.show()\n";
    viz_script.close();
    
    cout << "\nVisualization script created: visualize_bitcoin.py\n";
    cout << "Run 'python visualize_bitcoin.py' to generate plots\n";
    
    return 0;
}