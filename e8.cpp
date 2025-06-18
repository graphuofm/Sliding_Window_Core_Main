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
#include <omp.h>
#include <dirent.h>
#include <sys/stat.h>

using namespace std;
using vertex_t = int;
using edge_t = pair<vertex_t, vertex_t>;
using timestamp_t = long long;

// 全局结果文件
ofstream global_results;

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

// UCR并行版本 - 可配置线程数
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
    UCRParallel(Graph& g, int threads = 4) : G(g), num_threads(threads) {
        omp_set_num_threads(num_threads);
        reset();
    }
    
    void reset() {
        size_t n = G.num_vertices;
        r_values.clear();
        s_values.clear();
        r_values.resize(n, 0);
        s_values.resize(n, 0);
        
        // 并行计算r值
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
        
        // 并行计算s值
        #pragma omp parallel for num_threads(num_threads)
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
        }
        if (max_vertex_id >= r_values.size()) {
            r_values.resize(max_vertex_id + 1, 0);
            s_values.resize(max_vertex_id + 1, 0);
        }
        
        // Phase 1: 批量删除边
        unordered_map<int, vector<vertex_t>> k_groups_remove;
        
        for (const auto& edge : remove_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u < G.num_vertices && v < G.num_vertices) {
                int ku = G.core[u];
                int kv = G.core[v];
                
                if (ku < kv && u < r_values.size()) r_values[u]--;
                else if (ku > kv && v < r_values.size()) r_values[v]--;
                
                if (ku > 0) k_groups_remove[ku].push_back(u);
                if (kv > 0) k_groups_remove[kv].push_back(v);
                
                G.remove_edge(u, v);
            }
        }
        
        // 并行处理各个k层级的降级
        for (auto& pair : k_groups_remove) {
            int k = pair.first;
            vector<vertex_t>& vertices = pair.second;
            
            #pragma omp parallel for num_threads(num_threads)
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
            
            if (ku < kv && u < r_values.size()) r_values[u]++;
            else if (ku > kv && v < r_values.size()) r_values[v]++;
            
            affected_vertices.push_back(u);
            affected_vertices.push_back(v);
        }
        
        // 并行批量更新s值
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < affected_vertices.size(); ++i) {
            vertex_t v = affected_vertices[i];
            update_s_value(v);
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] == G.core[v]) {
                    update_s_value(w);
                }
            }
        }
        
        // 并行批量处理核心度升级
        #pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < affected_vertices.size(); ++i) {
            vertex_t v = affected_vertices[i];
            if (v < r_values.size() && r_values[v] + s_values[v] > G.core[v]) {
                G.core[v]++;
            }
        }
        
        return timer.elapsed_milliseconds();
    }
};

// 并行性能实验类
class ParallelPerformanceExperiment {
private:
    string dataset_path;
    string dataset_name;
    
public:
    ParallelPerformanceExperiment(const string& path, const string& name) 
        : dataset_path(path), dataset_name(name) {}
    
    void run() {
        LOG("=====================================");
        LOG("Dataset: " + dataset_name);
        LOG("=====================================");
        
        // 检查文件大小
        struct stat st;
        if (stat(dataset_path.c_str(), &st) == 0) {
            size_t file_size_gb = st.st_size / (1024LL * 1024LL * 1024LL);
            LOG("File size: " + to_string(file_size_gb) + " GB");
            
            if (file_size_gb > 20) {
                LOG("SKIPPING: File too large");
                return;
            }
        }
        
        // 使用传统方式接收返回值
        timestamp_t min_ts, max_ts;
        size_t total_edges;
        tie(min_ts, max_ts, total_edges) = scan_timestamp_range();
        
        if (total_edges == 0) {
            LOG("ERROR: No valid edges found");
            return;
        }
        
        LOG("Time range: " + to_string(min_ts) + " to " + to_string(max_ts));
        LOG("Total edges: " + to_string(total_edges));
        
        // 测试不同的窗口大小和线程数
        vector<int> window_sizes = {50, 100};
        vector<int> thread_counts = {1, 2, 4, 8};
        
        for (int window_size : window_sizes) {
            run_thread_experiment(window_size, thread_counts, min_ts, max_ts);
        }
    }
    
private:
    tuple<timestamp_t, timestamp_t, size_t> scan_timestamp_range() {
        LOG("Scanning timestamp range...");
        
        ifstream file(dataset_path);
        if (!file.is_open()) {
            return make_tuple(0, 0, 0);
        }
        
        string line;
        timestamp_t min_ts = LLONG_MAX;
        timestamp_t max_ts = LLONG_MIN;
        size_t edge_count = 0;
        size_t line_count = 0;
        
        Timer timer;
        
        while (getline(file, line)) {
            line_count++;
            if (line_count % 10000000 == 0) {
                LOG("  Scanned " + to_string(line_count) + " lines, found " + 
                    to_string(edge_count) + " valid edges");
                
                if (timer.elapsed_seconds() > 1800) {
                    LOG("  TIMEOUT: Scanning taking too long, stopping");
                    break;
                }
            }
            
            if (line.empty() || line[0] == '%' || line[0] == '#') continue;
            
            istringstream iss(line);
            vertex_t src, dst;
            timestamp_t ts;
            
            if (iss >> src >> dst >> ts && src != dst) {
                min_ts = min(min_ts, ts);
                max_ts = max(max_ts, ts);
                edge_count++;
            }
        }
        
        file.close();
        LOG("Scan complete: " + to_string(edge_count) + " edges in " + 
            to_string(timer.elapsed_seconds()) + " seconds");
        
        return make_tuple(min_ts, max_ts, edge_count);
    }
    
    void run_thread_experiment(int window_size, const vector<int>& thread_counts,
                              timestamp_t min_ts, timestamp_t max_ts) {
        LOG("\n--- Window size: " + to_string(window_size) + " bins ---");
        
        const int total_bins = 1000;
        double bin_span = static_cast<double>(max_ts - min_ts + 1) / total_bins;
        
        int starting_bin = 400;
        if (starting_bin + window_size > total_bins) {
            starting_bin = total_bins - window_size - 10;
        }
        if (starting_bin < 0) starting_bin = 0;
        
        // 一次性读取并按bin分组所有边
        LOG("Loading and indexing all edges by bins...");
        vector<vector<edge_t>> bins(total_bins);
        
        ifstream file(dataset_path);
        if (!file.is_open()) {
            LOG("ERROR: Cannot open file for indexing");
            return;
        }
        
        string line;
        size_t total_loaded = 0;
        Timer load_timer;
        
        while (getline(file, line)) {
            if (line.empty() || line[0] == '%' || line[0] == '#') continue;
            
            istringstream iss(line);
            vertex_t src, dst;
            timestamp_t ts;
            
            if (iss >> src >> dst >> ts && src != dst) {
                int bin_idx = min(static_cast<int>((ts - min_ts) / bin_span), total_bins - 1);
                bins[bin_idx].push_back(make_pair(src, dst));
                total_loaded++;
                
                if (total_loaded % 5000000 == 0) {
                    LOG("  Indexed " + to_string(total_loaded) + " edges...");
                    
                    if (load_timer.elapsed_seconds() > 1200) {
                        LOG("  TIMEOUT: Loading taking too long, stopping");
                        break;
                    }
                }
            }
        }
        file.close();
        
        LOG("Indexing complete: " + to_string(total_loaded) + " edges in " + 
            to_string(load_timer.elapsed_seconds()) + " seconds");
        
        // 构建初始窗口
        LOG("Building initial window (bins " + to_string(starting_bin) + " to " + 
            to_string(starting_bin + window_size - 1) + ")...");
        
        Graph initial_graph;
        size_t initial_edges = 0;
        
        for (int i = starting_bin; i < starting_bin + window_size; ++i) {
            initial_edges += bins[i].size();
            for (const auto& edge : bins[i]) {
                initial_graph.add_edge(edge.first, edge.second);
            }
        }
        
        LOG("  Initial edges in window: " + to_string(initial_edges));
        
        if (initial_edges > 20000000) {
            LOG("  SKIPPING: Too many edges in window");
            initial_graph.clear();
            return;
        }
        
        // 计算初始核心度
        Timer init_timer;
        LOG("  Computing initial core numbers...");
        bool success = initial_graph.compute_core_numbers_bz(600.0);
        
        if (!success) {
            LOG("  SKIPPING: Initial core computation timeout");
            initial_graph.clear();
            return;
        }
        
        double init_time = init_timer.elapsed_milliseconds();
        LOG("  Initial core computation: " + to_string(init_time) + " ms");
        
        // 进行滑动测试
        int remove_bin = starting_bin;
        int add_bin = starting_bin + window_size;
        
        if (add_bin >= total_bins) {
            LOG("No bins available for sliding");
            initial_graph.clear();
            return;
        }
        
        LOG("\nSlide test:");
        LOG("  Remove: " + to_string(bins[remove_bin].size()) + " edges");
        LOG("  Add: " + to_string(bins[add_bin].size()) + " edges");
        
        // 测试不同线程数
        vector<pair<int, double>> results;
        
        for (int threads : thread_counts) {
            LOG("\n  Testing with " + to_string(threads) + " threads:");
            
            // 运行多次取平均值
            const int num_runs = 3;
            double total_time = 0;
            
            for (int run = 0; run < num_runs; ++run) {
                Graph ucr_graph = initial_graph.copy();
                UCRParallel ucr_core(ucr_graph, threads);
                
                Timer ucr_timer;
                double ucr_time = ucr_core.process_sliding_window(bins[remove_bin], bins[add_bin]);
                total_time += ucr_time;
                
                LOG("    Run " + to_string(run + 1) + ": " + to_string(ucr_time) + " ms");
                
                ucr_graph.clear();
            }
            
            double avg_time = total_time / num_runs;
            results.push_back(make_pair(threads, avg_time));
            
            LOG("    Average: " + to_string(avg_time) + " ms");
            
            // 写入结果
            global_results << dataset_name << "\t"
                          << window_size << "\t"
                          << threads << "\t"
                          << fixed << setprecision(2)
                          << avg_time << "\n";
            global_results.flush();
        }
        
        // 计算加速比
        if (!results.empty()) {
            double base_time = results[0].second; // 1线程的时间
            LOG("\n  Speedup analysis:");
            for (size_t i = 0; i < results.size(); ++i) {
                int threads = results[i].first;
                double time = results[i].second;
                double speedup = base_time / time;
                LOG("    " + to_string(threads) + " threads: " + 
                    to_string(speedup) + "x speedup");
            }
        }
        
        // 清理内存
        initial_graph.clear();
        for (auto& bin : bins) {
            vector<edge_t>().swap(bin);
        }
        vector<vector<edge_t>>().swap(bins);
        
        LOG("  Memory cleaned up");
    }
};

// 获取数据集文件列表
vector<pair<string, string>> get_datasets(const string& dir) {
    vector<pair<string, string>> datasets;
    
    DIR* dirp = opendir(dir.c_str());
    if (dirp == nullptr) {
        LOG("ERROR: Failed to open directory: " + dir);
        return datasets;
    }
    
    struct dirent* dp;
    while ((dp = readdir(dirp)) != nullptr) {
        string filename(dp->d_name);
        if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".txt") {
            string filepath = dir + "/" + filename;
            string name = filename.substr(0, filename.size() - 4);
            
            if (name.substr(0, 2) != "._") {
                datasets.push_back(make_pair(filepath, name));
            }
        }
    }
    closedir(dirp);
    
    // 选择代表性数据集进行并行测试
    vector<pair<string, string>> selected;
    for (const auto& ds : datasets) {
        // 选择中等规模的数据集进行并行测试
        if (ds.second == "sx-superuser" || 
            ds.second == "wiki-talk-temporal" ||
            ds.second == "cit-Patents" ||
            ds.second == "soc-LiveJournal1") {
            selected.push_back(ds);
        }
    }
    
    return selected;
}

int main() {
    LOG("===========================================");
    LOG("Experiment 8: UCR Parallel Performance Analysis");
    LOG("===========================================");
    
    // 打开全局结果文件
    string output_file = "e8_parallel_results.txt";
    global_results.open(output_file);
    
    if (!global_results.is_open()) {
        LOG("ERROR: Failed to open output file: " + output_file);
        return 1;
    }
    
    // 写入文件头
    global_results << "Dataset\tWindow_Size\tThreads\tTime(ms)\n";
    global_results.flush();
    
    LOG("Results will be saved to: " + output_file);
    
    // 获取选定的数据集
    string dataset_dir = "dataset";
    auto datasets = get_datasets(dataset_dir);
    
    LOG("\nSelected " + to_string(datasets.size()) + " datasets for parallel testing");
    
    // 对每个数据集运行实验
    int dataset_count = 0;
    for (size_t i = 0; i < datasets.size(); ++i) {
        const string& filepath = datasets[i].first;
        const string& name = datasets[i].second;
        
        dataset_count++;
        LOG("\n[" + to_string(dataset_count) + "/" + to_string(datasets.size()) + "] " + name);
        
        try {
            ParallelPerformanceExperiment experiment(filepath, name);
            experiment.run();
        } catch (const exception& e) {
            LOG("ERROR: Exception processing " + name + ": " + string(e.what()));
        } catch (...) {
            LOG("ERROR: Unknown exception processing " + name);
        }
        
        global_results.flush();
    }
    
    global_results.close();
    
    LOG("\n===========================================");
    LOG("Experiment 8 completed!");
    LOG("Results saved to: " + output_file);
    LOG("===========================================");
    
    return 0;
}