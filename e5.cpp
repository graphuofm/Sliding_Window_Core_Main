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
#include <set>
#include <random>
#include <omp.h>
#include <dirent.h>
#include <sys/stat.h>

using namespace std;
using vertex_t = int;
using edge_t = pair<vertex_t, vertex_t>;
using timestamp_t = long long;

// 全局结果文件
ofstream global_results;

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

// 获取当前时间字符串
string get_current_time() {
    auto now = chrono::system_clock::now();
    auto time_t = chrono::system_clock::to_time_t(now);
    char buffer[100];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime(&time_t));
    return string(buffer);
}

// 日志输出
#define LOG(msg) do { \
    cout << "[" << get_current_time() << "] " << msg << endl; \
    cout.flush(); \
} while(0)

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
    
    // BZ算法计算核心度
    bool compute_core_numbers_bz(double timeout_seconds = 600.0) {
        Timer timer;
        
        if (num_vertices == 0) return true;
        
        fill(core.begin(), core.end(), 0);
        
        vector<int> degree(num_vertices);
        int max_degree = 0;
        for (vertex_t v = 0; v < num_vertices; ++v) {
            degree[v] = adj[v].size();
            max_degree = max(max_degree, degree[v]);
        }
        
        if (max_degree == 0) return true;
        
        vector<vector<vertex_t>> bins(max_degree + 1);
        for (vertex_t v = 0; v < num_vertices; ++v) {
            bins[degree[v]].push_back(v);
        }
        
        vector<bool> processed(num_vertices, false);
        
        for (int d = 0; d <= max_degree; ++d) {
            if (d % 1000 == 0 && timer.elapsed_seconds() > timeout_seconds) {
                LOG("    BZ computation timeout after " + to_string(timer.elapsed_seconds()) + " seconds");
                return false;
            }
            
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
        
        return true;
    }
    
    Graph copy() const {
        Graph g;
        g.num_vertices = num_vertices;
        g.adj = adj;
        g.core = core;
        return g;
    }
    
    void clear() {
        for (auto& neighbors : adj) {
            vector<vertex_t>().swap(neighbors);
        }
        vector<vector<vertex_t>>().swap(adj);
        vector<int>().swap(core);
        num_vertices = 0;
    }
};

// UCR并行版本 - 改进版
class UCRParallel {
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
    UCRParallel(Graph& g) : G(g) {
        reset();
    }
    
    void reset() {
        size_t n = G.num_vertices;
        r_values.clear();
        s_values.clear();
        r_values.resize(n, 0);
        s_values.resize(n, 0);
        
        // 并行计算r值
        #pragma omp parallel for schedule(dynamic, 1000)
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
        
        // 并行计算s值
        #pragma omp parallel for schedule(dynamic, 1000)
        for (vertex_t v = 0; v < n; ++v) {
            update_s_value(v);
        }
    }
    
    // 改进的并行处理版本
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
        }
        if (max_vertex_id >= r_values.size()) {
            r_values.resize(max_vertex_id + 1, 0);
            s_values.resize(max_vertex_id + 1, 0);
        }
        
        // Phase 1: 批量删除边并更新r值
        unordered_map<int, vector<vertex_t>> k_groups_remove;
        
        for (const auto& edge : remove_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u < G.num_vertices && v < G.num_vertices) {
                int ku = G.core[u];
                int kv = G.core[v];
                
                if (ku < kv && u < r_values.size()) {
                    #pragma omp atomic
                    r_values[u]--;
                }
                else if (ku > kv && v < r_values.size()) {
                    #pragma omp atomic
                    r_values[v]--;
                }
                
                if (ku > 0) k_groups_remove[ku].push_back(u);
                if (kv > 0) k_groups_remove[kv].push_back(v);
                
                G.remove_edge(u, v);
            }
        }
        
        // 并行处理核心度降级
        vector<int> k_values;
        for (const auto& pair : k_groups_remove) {
            k_values.push_back(pair.first);
        }
        sort(k_values.begin(), k_values.end());
        
        for (int k : k_values) {
            auto& vertices = k_groups_remove[k];
            
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < vertices.size(); ++i) {
                vertex_t v = vertices[i];
                if (G.core[v] == k && r_values[v] + s_values[v] < k) {
                    G.core[v] = k - 1;
                }
            }
        }
        
        // Phase 2: 批量添加边
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
            
            if (ku < kv && u < r_values.size()) {
                #pragma omp atomic
                r_values[u]++;
            }
            else if (ku > kv && v < r_values.size()) {
                #pragma omp atomic
                r_values[v]++;
            }
            
            affected_vertices.push_back(u);
            affected_vertices.push_back(v);
        }
        
        // 去重
        sort(affected_vertices.begin(), affected_vertices.end());
        affected_vertices.erase(unique(affected_vertices.begin(), affected_vertices.end()), 
                               affected_vertices.end());
        
        // 并行更新s值 - 使用更细粒度的并行
        #pragma omp parallel for schedule(dynamic, 100)
        for (size_t i = 0; i < affected_vertices.size(); ++i) {
            vertex_t v = affected_vertices[i];
            update_s_value(v);
            
            // 更新邻居的s值
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] == G.core[v]) {
                    update_s_value(w);
                }
            }
        }
        
        // 并行处理核心度升级 - 分组处理避免竞争
        unordered_map<int, vector<vertex_t>> upgrade_groups;
        for (vertex_t v : affected_vertices) {
            if (v < r_values.size() && r_values[v] + s_values[v] > G.core[v]) {
                upgrade_groups[G.core[v]].push_back(v);
            }
        }
        
        for (auto& [k, vertices] : upgrade_groups) {
            #pragma omp parallel for
            for (size_t i = 0; i < vertices.size(); ++i) {
                vertex_t v = vertices[i];
                if (r_values[v] + s_values[v] > G.core[v]) {
                    G.core[v]++;
                }
            }
        }
        
        return timer.elapsed_milliseconds();
    }
};

// 窗口大小和步长实验类 - 使用UCR-Parallel
class WindowStepExperimentParallel {
private:
    string dataset_path;
    string dataset_name;
    
public:
    WindowStepExperimentParallel(const string& path, const string& name) 
        : dataset_path(path), dataset_name(name) {}
    
    void run() {
        LOG("=====================================");
        LOG("Dataset: " + dataset_name);
        LOG("Window Size and Step Size Analysis with UCR-Parallel");
        LOG("=====================================");
        
        // 读取时间戳范围
        auto [min_ts, max_ts, total_edges] = scan_timestamp_range();
        if (total_edges == 0) {
            LOG("ERROR: No valid edges found");
            return;
        }
        
        LOG("Time range: " + to_string(min_ts) + " to " + to_string(max_ts));
        LOG("Total edges: " + to_string(total_edges));
        
        // 一次性加载所有数据到bins
        const int total_bins = 1000;
        double bin_span = static_cast<double>(max_ts - min_ts + 1) / total_bins;
        
        LOG("\nLoading all edges into bins...");
        vector<vector<edge_t>> bins(total_bins);
        
        ifstream file(dataset_path);
        if (!file.is_open()) {
            LOG("ERROR: Cannot open file");
            return;
        }
        
        string line;
        size_t loaded = 0;
        Timer load_timer;
        
        while (getline(file, line)) {
            if (line.empty() || line[0] == '%' || line[0] == '#') continue;
            
            istringstream iss(line);
            vertex_t src, dst;
            timestamp_t ts;
            
            if (iss >> src >> dst >> ts && src != dst) {
                int bin_idx = min(static_cast<int>((ts - min_ts) / bin_span), total_bins - 1);
                bins[bin_idx].push_back({src, dst});
                loaded++;
                
                if (loaded % 5000000 == 0) {
                    LOG("  Loaded " + to_string(loaded) + " edges...");
                }
            }
        }
        file.close();
        
        LOG("Loading complete: " + to_string(loaded) + " edges in " + 
            to_string(load_timer.elapsed_seconds()) + " seconds");
        
        // 定义测试参数 - 缩小范围，专注于问题区域
        vector<int> window_sizes = {10, 20, 30, 50, 80, 100, 120, 150};
        vector<int> step_sizes = {1, 2, 5, 10, 20, 30, 50};
        
        // 写入结果文件头
        global_results << "Dataset\tWindowSize\tStepSize\tRemoveEdges\tAddEdges\t"
                      << "BZ_Time(ms)\tUCR_Parallel_Time(ms)\tSpeedup\t"
                      << "BZ_EdgePerMs\tUCR_EdgePerMs\tThreads\n";
        global_results.flush();
        
        // 对每个窗口大小和步长组合进行测试
        int experiment_count = 0;
        int total_experiments = window_sizes.size() * step_sizes.size();
        
        for (int window_size : window_sizes) {
            for (int step_size : step_sizes) {
                // 跳过步长大于窗口大小的情况
                if (step_size > window_size) continue;
                
                experiment_count++;
                LOG("\n[" + to_string(experiment_count) + "/" + to_string(total_experiments) + "] "
                    + "Window=" + to_string(window_size) + " bins, Step=" + to_string(step_size) + " bins");
                
                // 选择起始位置
                int start_bin = 400;
                if (start_bin + window_size + step_size >= total_bins) {
                    start_bin = total_bins - window_size - step_size - 10;
                }
                if (start_bin < 0) {
                    LOG("  SKIP: Not enough bins for this configuration");
                    continue;
                }
                
                // 构建初始窗口
                LOG("  Building initial window...");
                Graph initial_graph;
                size_t initial_edges = 0;
                
                for (int i = start_bin; i < start_bin + window_size; ++i) {
                    initial_edges += bins[i].size();
                    for (const auto& edge : bins[i]) {
                        initial_graph.add_edge(edge.first, edge.second);
                    }
                }
                
                LOG("  Initial edges: " + to_string(initial_edges));
                LOG("  Initial vertices: " + to_string(initial_graph.num_vertices));
                
                // 计算初始核心度
                Timer init_timer;
                bool success = initial_graph.compute_core_numbers_bz(300.0);
                if (!success) {
                    LOG("  SKIP: Initial core computation timeout");
                    initial_graph.clear();
                    continue;
                }
                double init_time = init_timer.elapsed_milliseconds();
                LOG("  Initial core computation: " + to_string(init_time) + " ms");
                
                // 收集要删除和添加的边
                vector<edge_t> remove_edges;
                vector<edge_t> add_edges;
                
                // 删除的边：从start_bin到start_bin+step_size-1
                for (int i = start_bin; i < start_bin + step_size && i < start_bin + window_size; ++i) {
                    for (const auto& edge : bins[i]) {
                        remove_edges.push_back(edge);
                    }
                }
                
                // 添加的边：从start_bin+window_size到start_bin+window_size+step_size-1
                for (int i = start_bin + window_size; i < start_bin + window_size + step_size && i < total_bins; ++i) {
                    for (const auto& edge : bins[i]) {
                        add_edges.push_back(edge);
                    }
                }
                
                LOG("  Remove edges: " + to_string(remove_edges.size()));
                LOG("  Add edges: " + to_string(add_edges.size()));
                
                size_t total_edge_updates = remove_edges.size() + add_edges.size();
                
                // 测试BZ重计算
                LOG("  Testing BZ recomputation...");
                Timer bz_timer;
                Graph bz_graph;
                
                // 构建滑动后的窗口
                for (int i = start_bin + step_size; i < start_bin + window_size + step_size && i < total_bins; ++i) {
                    for (const auto& edge : bins[i]) {
                        bz_graph.add_edge(edge.first, edge.second);
                    }
                }
                
                bool bz_success = bz_graph.compute_core_numbers_bz(300.0);
                if (!bz_success) {
                    LOG("    BZ timeout");
                    bz_graph.clear();
                    initial_graph.clear();
                    continue;
                }
                
                double bz_time = bz_timer.elapsed_milliseconds();
                double bz_edge_per_ms = total_edge_updates / bz_time;
                LOG("    BZ time: " + to_string(bz_time) + " ms (" + 
                    to_string(bz_edge_per_ms) + " edges/ms)");
                
                bz_graph.clear();
                
                // 测试UCR-Parallel
                LOG("  Testing UCR-Parallel with " + to_string(omp_get_max_threads()) + " threads...");
                Graph ucr_graph = initial_graph.copy();
                UCRParallel ucr_core(ucr_graph);
                
                double ucr_time = ucr_core.process_sliding_window(remove_edges, add_edges);
                double ucr_edge_per_ms = total_edge_updates / ucr_time;
                double speedup = bz_time / ucr_time;
                
                LOG("    UCR-Parallel time: " + to_string(ucr_time) + " ms (" + 
                    to_string(ucr_edge_per_ms) + " edges/ms)");
                LOG("    Speedup: " + to_string(speedup) + "x");
                
                // 写入结果
                global_results << dataset_name << "\t"
                             << window_size << "\t"
                             << step_size << "\t"
                             << remove_edges.size() << "\t"
                             << add_edges.size() << "\t"
                             << fixed << setprecision(2)
                             << bz_time << "\t"
                             << ucr_time << "\t"
                             << speedup << "\t"
                             << bz_edge_per_ms << "\t"
                             << ucr_edge_per_ms << "\t"
                             << omp_get_max_threads() << "\n";
                global_results.flush();
                
                // 清理
                ucr_graph.clear();
                initial_graph.clear();
                
                // 每10个实验后进行内存清理
                if (experiment_count % 10 == 0) {
                    LOG("  Performing memory cleanup...");
                }
            }
        }
        
        // 最终清理
        for (auto& bin : bins) {
            vector<edge_t>().swap(bin);
        }
        vector<vector<edge_t>>().swap(bins);
        
        LOG("\nExperiment completed!");
    }
    
private:
    tuple<timestamp_t, timestamp_t, size_t> scan_timestamp_range() {
        LOG("Scanning timestamp range...");
        
        ifstream file(dataset_path);
        if (!file.is_open()) {
            return {0, 0, 0};
        }
        
        string line;
        timestamp_t min_ts = LLONG_MAX;
        timestamp_t max_ts = LLONG_MIN;
        size_t edge_count = 0;
        
        Timer timer;
        
        while (getline(file, line)) {
            if (line.empty() || line[0] == '%' || line[0] == '#') continue;
            
            istringstream iss(line);
            vertex_t src, dst;
            timestamp_t ts;
            
            if (iss >> src >> dst >> ts && src != dst) {
                min_ts = min(min_ts, ts);
                max_ts = max(max_ts, ts);
                edge_count++;
                
                if (edge_count % 10000000 == 0) {
                    LOG("  Scanned " + to_string(edge_count) + " edges...");
                }
            }
        }
        
        file.close();
        LOG("Scan complete: " + to_string(edge_count) + " edges");
        
        return {min_ts, max_ts, edge_count};
    }
};

int main() {
    LOG("===========================================");
    LOG("Experiment 5 Parallel: Window Size and Step Size Analysis");
    LOG("Using UCR-Parallel Algorithm");
    LOG("Target: sx-stackoverflow dataset");
    LOG("===========================================");
    
    // 设置OpenMP线程数 - 使用8个线程
    omp_set_num_threads(8);
    LOG("OpenMP threads set to: " + to_string(omp_get_max_threads()));
    
    // 打开结果文件
    string output_file = "e5_parallel_results.txt";
    global_results.open(output_file);
    
    if (!global_results.is_open()) {
        LOG("ERROR: Failed to open output file: " + output_file);
        return 1;
    }
    
    LOG("Results will be saved to: " + output_file);
    
    // 运行sx-stackoverflow数据集实验
    string dataset_path = "dataset/sx-stackoverflow.txt";
    string dataset_name = "sx-stackoverflow";
    
    // 检查文件是否存在
    ifstream test_file(dataset_path);
    if (!test_file.is_open()) {
        LOG("ERROR: Cannot find dataset file: " + dataset_path);
        global_results.close();
        return 1;
    }
    test_file.close();
    
    try {
        WindowStepExperimentParallel experiment(dataset_path, dataset_name);
        experiment.run();
    } catch (const exception& e) {
        LOG("ERROR: Exception - " + string(e.what()));
    } catch (...) {
        LOG("ERROR: Unknown exception");
    }
    
    global_results.close();
    
    LOG("\n===========================================");
    LOG("Experiment 5 Parallel completed!");
    LOG("Results saved to: " + output_file);
    LOG("===========================================");
    
    return 0;
}