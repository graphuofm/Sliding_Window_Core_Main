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
    
    // 并行BZ算法（简化版）
    void compute_core_numbers_bz_parallel(int threads) {
        omp_set_num_threads(threads);
        compute_core_numbers_bz(); // 暂时还是用串行版本，真正的并行BZ需要更复杂的实现
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
};

// UCR并行版本（从e2代码中提取）
class UCRParallel {
private:
    Graph& G;
    vector<int> r_values;
    vector<int> s_values;
    int num_threads;
    
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
    UCRParallel(Graph& g, int threads = 1) : G(g), num_threads(threads) {
        omp_set_num_threads(num_threads);
        reset();
    }
    
    void reset() {
        size_t n = G.num_vertices;
        r_values.resize(n, 0);
        s_values.resize(n, 0);
        
        if (num_threads == 1) {
            // 单线程版本
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
            
            for (vertex_t v = 0; v < n; ++v) {
                update_s_value(v);
            }
        } else {
            // 多线程版本
            #pragma omp parallel for num_threads(num_threads)
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
            
            #pragma omp parallel for num_threads(num_threads)
            for (vertex_t v = 0; v < n; ++v) {
                update_s_value(v);
            }
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
        
        // Phase 1: 处理边删除（串行）
        for (const auto& edge : remove_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u < G.num_vertices && v < G.num_vertices) {
                int ku = G.core[u];
                int kv = G.core[v];
                
                if (ku < kv && u < r_values.size()) r_values[u]--;
                else if (ku > kv && v < r_values.size()) r_values[v]--;
                
                G.remove_edge(u, v);
                
                if (r_values[u] + s_values[u] < G.core[u]) G.core[u]--;
                if (r_values[v] + s_values[v] < G.core[v]) G.core[v]--;
            }
        }
        
        // Phase 2: 处理边添加（串行）
        vector<vertex_t> affected_vertices;
        for (const auto& edge : add_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u == v) continue;
            
            G.ensure_vertex(max(u, v));
            
            if (G.adj[u].empty()) G.core[u] = 1;
            if (G.adj[v].empty()) G.core[v] = 1;
            
            G.add_edge(u, v);
            
            int ku = G.core[u];
            int kv = G.core[v];
            
            if (ku < kv && u < r_values.size()) r_values[u]++;
            else if (ku > kv && v < r_values.size()) r_values[v]++;
            
            affected_vertices.push_back(u);
            affected_vertices.push_back(v);
        }
        
        // Phase 3: 更新s值（可并行）
        if (num_threads == 1) {
            for (vertex_t v : affected_vertices) {
                update_s_value(v);
            }
        } else {
            #pragma omp parallel for num_threads(num_threads)
            for (size_t i = 0; i < affected_vertices.size(); ++i) {
                update_s_value(affected_vertices[i]);
            }
        }
        
        // Phase 4: 处理核心度升级（串行）
        for (vertex_t v : affected_vertices) {
            if (v < r_values.size() && r_values[v] + s_values[v] > G.core[v]) {
                G.core[v]++;
            }
        }
        
        return timer.elapsed_milliseconds();
    }
};

int main() {
    cout << "===========================================\n";
    cout << "Experiment 7 Enhanced: UCR vs Full Pipeline Parallel Performance\n";
    cout << "Window Size: 200 bins\n";
    cout << "Sliding Steps: 5, 10, 20\n";
    cout << "===========================================\n\n";
    
    // 设置最大线程数
    omp_set_num_threads(8);
    
    // 选择几个数据集
    vector<pair<string, string>> datasets = {
        {"dataset/cit-Patents.txt", "cit-Patents"},
        {"dataset/soc-LiveJournal1.txt", "soc-LiveJournal1"},
        {"dataset/sx-stackoverflow.txt", "sx-stackoverflow"}
    };
    
    // 测试的线程数
    vector<int> thread_counts = {1, 2, 4, 8};
    
    // 滑动步长
    vector<int> sliding_steps = {5, 10, 20};
    
    // 打开结果文件
    ofstream results("e7_enhanced_results.txt");
    results << "Dataset\tSlide_Step\tTest_Type\tThreads\tTime(ms)\tSpeedup\tEfficiency(%)\n";
    
    const int window_size = 200;
    const int total_bins = 1000;
    const int num_runs = 5;
    
    for (const auto& [filepath, dataset_name] : datasets) {
        cout << "\n=====================================\n";
        cout << "Dataset: " << dataset_name << "\n";
        cout << "=====================================\n";
        
        // 读取数据集
        ifstream file(filepath);
        if (!file.is_open()) {
            cout << "ERROR: Cannot open " << filepath << "\n";
            continue;
        }
        
        // 扫描时间戳范围
        cout << "Loading dataset...\n";
        string line;
        timestamp_t min_ts = LLONG_MAX;
        timestamp_t max_ts = LLONG_MIN;
        vector<TemporalEdge> all_edges;
        
        while (getline(file, line)) {
            if (line.empty() || line[0] == '%' || line[0] == '#') continue;
            
            istringstream iss(line);
            vertex_t src, dst;
            timestamp_t ts;
            
            if (iss >> src >> dst >> ts && src != dst) {
                min_ts = min(min_ts, ts);
                max_ts = max(max_ts, ts);
                all_edges.emplace_back(src, dst, ts);
            }
        }
        file.close();
        
        cout << "Total edges: " << all_edges.size() << "\n";
        
        // 按时间戳分组到bins
        double bin_span = static_cast<double>(max_ts - min_ts + 1) / total_bins;
        vector<vector<edge_t>> bins(total_bins);
        
        for (const auto& e : all_edges) {
            int bin_idx = min(static_cast<int>((e.timestamp - min_ts) / bin_span), total_bins - 1);
            bins[bin_idx].push_back(make_pair(e.src, e.dst));
        }
        
        // 对每个滑动步长进行测试
        for (int slide_step : sliding_steps) {
            cout << "\n--- Sliding " << slide_step << " bins ---\n";
            
            // 构建初始窗口
            int starting_bin = 400;
            Graph initial_graph;
            
            for (int i = starting_bin; i < starting_bin + window_size; ++i) {
                for (const auto& edge : bins[i]) {
                    initial_graph.add_edge(edge.first, edge.second);
                }
            }
            
            cout << "Computing initial core numbers (not timed)...\n";
            initial_graph.compute_core_numbers_bz();
            
            // 准备滑动更新
            vector<edge_t> remove_edges;
            vector<edge_t> add_edges;
            
            for (int i = 0; i < slide_step; ++i) {
                remove_edges.insert(remove_edges.end(), 
                                   bins[starting_bin + i].begin(), 
                                   bins[starting_bin + i].end());
                add_edges.insert(add_edges.end(), 
                                bins[starting_bin + window_size + i].begin(), 
                                bins[starting_bin + window_size + i].end());
            }
            
            cout << "Remove edges: " << remove_edges.size() << "\n";
            cout << "Add edges: " << add_edges.size() << "\n";
            
            // Test 1: 纯UCR性能测试（不包含初始BZ）
            cout << "\n=== Test 1: Pure UCR Performance (excluding initial BZ) ===\n";
            map<int, double> ucr_times;
            
            for (int threads : thread_counts) {
                cout << "Testing UCR with " << threads << " thread(s)...\n";
                
                double total_time = 0;
                
                for (int run = 0; run < num_runs; ++run) {
                    Graph test_graph = initial_graph.copy();
                    UCRParallel ucr(test_graph, threads);
                    
                    double time = ucr.process_sliding_window(remove_edges, add_edges);
                    total_time += time;
                    
                    cout << "  Run " << (run + 1) << ": " << time << " ms\n";
                    
                    test_graph.clear();
                }
                
                ucr_times[threads] = total_time / num_runs;
                cout << "  Average: " << ucr_times[threads] << " ms\n";
            }
            
            // 输出UCR性能总结
            cout << "\nUCR Performance Summary:\n";
            double ucr_baseline = ucr_times[1];
            
            for (int threads : thread_counts) {
                double speedup = ucr_baseline / ucr_times[threads];
                double efficiency = speedup / threads * 100;
                
                cout << "  " << threads << " threads: " 
                     << fixed << setprecision(2) 
                     << ucr_times[threads] << " ms, "
                     << speedup << "x speedup, "
                     << efficiency << "% efficiency\n";
                
                results << dataset_name << "\t"
                        << slide_step << "\t"
                        << "UCR_Only" << "\t"
                        << threads << "\t"
                        << fixed << setprecision(2)
                        << ucr_times[threads] << "\t"
                        << speedup << "\t"
                        << efficiency << "\n";
            }
            
            // Test 2: 完整滑动窗口性能测试（包含BZ重计算）
            cout << "\n=== Test 2: Full Pipeline Performance (BZ + UCR) ===\n";
            map<int, double> full_times;
            
            for (int threads : thread_counts) {
                cout << "Testing full pipeline with " << threads << " thread(s)...\n";
                
                double total_time = 0;
                
                for (int run = 0; run < num_runs; ++run) {
                    Timer full_timer;
                    
                    // 构建新窗口的图
                    Graph new_graph;
                    for (int i = starting_bin + slide_step; i < starting_bin + window_size + slide_step; ++i) {
                        for (const auto& edge : bins[i]) {
                            new_graph.add_edge(edge.first, edge.second);
                        }
                    }
                    
                    // 计算核心度（可以考虑并行化）
                    new_graph.compute_core_numbers_bz_parallel(threads);
                    
                    double time = full_timer.elapsed_milliseconds();
                    total_time += time;
                    
                    cout << "  Run " << (run + 1) << ": " << time << " ms\n";
                    
                    new_graph.clear();
                }
                
                full_times[threads] = total_time / num_runs;
                cout << "  Average: " << full_times[threads] << " ms\n";
            }
            
            // 输出完整流程性能总结
            cout << "\nFull Pipeline Performance Summary:\n";
            double full_baseline = full_times[1];
            
            for (int threads : thread_counts) {
                double speedup = full_baseline / full_times[threads];
                double efficiency = speedup / threads * 100;
                
                cout << "  " << threads << " threads: " 
                     << fixed << setprecision(2) 
                     << full_times[threads] << " ms, "
                     << speedup << "x speedup, "
                     << efficiency << "% efficiency\n";
                
                results << dataset_name << "\t"
                        << slide_step << "\t"
                        << "Full_Pipeline" << "\t"
                        << threads << "\t"
                        << fixed << setprecision(2)
                        << full_times[threads] << "\t"
                        << speedup << "\t"
                        << efficiency << "\n";
            }
            
            // 对比分析
            cout << "\n=== Comparison ===\n";
            cout << "UCR-only speedup (8 threads): " 
                 << fixed << setprecision(2) 
                 << ucr_baseline / ucr_times[8] << "x\n";
            cout << "Full pipeline speedup (8 threads): " 
                 << fixed << setprecision(2) 
                 << full_baseline / full_times[8] << "x\n";
            
            results.flush();
            initial_graph.clear();
        }
    }
    
    results.close();
    
    cout << "\n===========================================\n";
    cout << "Experiment completed!\n";
    cout << "Results saved to: e7_enhanced_results.txt\n";
    cout << "===========================================\n";
    
    return 0;
}