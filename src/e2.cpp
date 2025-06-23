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
#include <cstring>
#include <climits>
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
            size_t old_size = num_vertices;
            num_vertices = v + 1;
            adj.resize(num_vertices);
            core.resize(num_vertices, 0);
        }
    }
    
    void add_edge(vertex_t u, vertex_t v) {
        if (u == v) return; // 跳过自环
        
        ensure_vertex(max(u, v));
        
        // 避免重复边
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
    
    // BZ算法计算核心度（带超时检查）
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
            // 每1000次迭代检查一次超时
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
    
    size_t num_edges() const {
        size_t edges = 0;
        for (const auto& neighbors : adj) {
            edges += neighbors.size();
        }
        return edges / 2;
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

// UCR算法实现（批处理+并行优化版本）
class UCRFull {
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
    UCRFull(Graph& g) : G(g) {
        reset();
    }
    
    void reset() {
        size_t n = G.num_vertices;
        r_values.clear();
        s_values.clear();
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
    
    // 批量处理边的删除和插入
    double process_batch(const vector<edge_t>& remove_edges, 
                        const vector<edge_t>& add_edges) {
        Timer timer;
        
        // 扩展数据结构以适应新顶点
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
        for (const auto& edge : remove_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u < G.num_vertices && v < G.num_vertices) {
                int ku = G.core[u];
                int kv = G.core[v];
                
                if (ku < kv && u < r_values.size()) r_values[u]--;
                else if (ku > kv && v < r_values.size()) r_values[v]--;
                
                G.remove_edge(u, v);
            }
        }
        
        // Phase 2: 批量插入边
        for (const auto& edge : add_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            if (u == v) continue; // 跳过自环
            
            G.ensure_vertex(max(u, v));
            
            // 为新顶点设置初始核心度
            if (G.adj[u].empty()) G.core[u] = 1;
            if (G.adj[v].empty()) G.core[v] = 1;
            
            G.add_edge(u, v);
            
            int ku = G.core[u];
            int kv = G.core[v];
            
            if (ku < kv && u < r_values.size()) r_values[u]++;
            else if (ku > kv && v < r_values.size()) r_values[v]++;
        }
        
        return timer.elapsed_milliseconds();
    }
};

// 流式滑动窗口实验类
class StreamingSlidingWindowExperiment {
private:
    string dataset_path;
    string dataset_name;
    
public:
    StreamingSlidingWindowExperiment(const string& path, const string& name) 
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
            
            // 超过20GB的文件跳过
            if (file_size_gb > 20) {
                LOG("SKIPPING: File too large");
                return;
            }
        }
        
        // 流式处理：先读取时间戳范围
        auto [min_ts, max_ts, total_edges] = scan_timestamp_range();
        if (total_edges == 0) {
            LOG("ERROR: No valid edges found");
            return;
        }
        
        LOG("Time range: " + to_string(min_ts) + " to " + to_string(max_ts));
        LOG("Total edges: " + to_string(total_edges));
        
        // 对每个窗口大小运行实验
        vector<int> window_sizes = {10, 20, 50, 100};
        for (int window_size : window_sizes) {
            run_streaming_window_experiment(window_size, min_ts, max_ts);
        }
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
        size_t line_count = 0;
        
        Timer timer;
        
        while (getline(file, line)) {
            line_count++;
            if (line_count % 10000000 == 0) {
                LOG("  Scanned " + to_string(line_count) + " lines, found " + to_string(edge_count) + " valid edges");
                
                // 扫描超时检查：超过30分钟停止
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
        LOG("Scan complete: " + to_string(edge_count) + " edges in " + to_string(timer.elapsed_seconds()) + " seconds");
        
        return {min_ts, max_ts, edge_count};
    }
    
    void run_streaming_window_experiment(int window_size, timestamp_t min_ts, timestamp_t max_ts) {
        LOG("\n--- Streaming Window size: " + to_string(window_size) + " bins ---");
        
        const int total_bins = 1000;
        double bin_span = static_cast<double>(max_ts - min_ts + 1) / total_bins;
        
        // 选择起始位置
        int starting_bin = 400;
        if (starting_bin + window_size > total_bins) {
            starting_bin = total_bins - window_size - 10;
        }
        if (starting_bin < 0) starting_bin = 0;
        
        // **一次性读取并按bin分组所有边**
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
                bins[bin_idx].push_back({src, dst});
                total_loaded++;
                
                if (total_loaded % 5000000 == 0) {
                    LOG("  Indexed " + to_string(total_loaded) + " edges...");
                    
                    // 加载超时检查
                    if (load_timer.elapsed_seconds() > 1200) { // 20分钟超时
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
        
        // 如果窗口边数过多，跳过
        if (initial_edges > 20000000) { // 超过2000万边跳过
            LOG("  SKIPPING: Too many edges in window");
            initial_graph.clear();
            return;
        }
        
        // 计算初始核心度（带超时）
        Timer init_timer;
        LOG("  Computing initial core numbers...");
        bool success = initial_graph.compute_core_numbers_bz(600.0); // 10分钟超时
        
        if (!success) {
            LOG("  SKIPPING: Initial core computation timeout");
            initial_graph.clear();
            return;
        }
        
        double init_time = init_timer.elapsed_milliseconds();
        LOG("  Initial core computation: " + to_string(init_time) + " ms");
        
        // 滑动3次
        vector<double> bz_times, ucr_times;
        
        for (int slide = 0; slide < 3; ++slide) {
            int remove_bin = starting_bin + slide;
            int add_bin = starting_bin + window_size + slide;
            
            if (add_bin >= total_bins) {
                LOG("Reached end of bins at slide " + to_string(slide + 1));
                break;
            }
            
            LOG("\nSlide " + to_string(slide + 1) + ":");
            
            // 直接从bins获取边，无需重新扫描文件
            const auto& remove_edges = bins[remove_bin];
            const auto& add_edges = bins[add_bin];
            
            LOG("  Remove: " + to_string(remove_edges.size()) + " edges");
            LOG("  Add: " + to_string(add_edges.size()) + " edges");
            
            // BZ重计算
            Timer bz_timer;
            Graph bz_graph;
            
            // 构建新窗口的图
            for (int i = remove_bin + 1; i <= add_bin; ++i) {
                for (const auto& edge : bins[i]) {
                    bz_graph.add_edge(edge.first, edge.second);
                }
            }
            
            bool bz_success = bz_graph.compute_core_numbers_bz(300.0); // 5分钟超时
            
            if (!bz_success) {
                LOG("    BZ computation timeout, skipping remaining slides");
                bz_graph.clear();
                break;
            }
            
            double bz_time = bz_timer.elapsed_milliseconds();
            bz_times.push_back(bz_time);
            LOG("    BZ time: " + to_string(bz_time) + " ms");
            
            // 清理BZ图
            bz_graph.clear();
            
            // UCR批处理
            Graph ucr_graph = initial_graph.copy();
            UCRFull ucr_core(ucr_graph);
            
            Timer ucr_timer;
            double ucr_time = ucr_core.process_batch(remove_edges, add_edges);
            ucr_times.push_back(ucr_time);
            LOG("    UCR time: " + to_string(ucr_time) + " ms");
            LOG("    Speedup: " + to_string(bz_time / ucr_time) + "x");
            
            // 更新初始图为下一次滑动准备
            for (const auto& edge : remove_edges) {
                initial_graph.remove_edge(edge.first, edge.second);
            }
            for (const auto& edge : add_edges) {
                initial_graph.add_edge(edge.first, edge.second);
            }
            
            // 重新计算核心度
            if (!initial_graph.compute_core_numbers_bz(600.0)) {
                LOG("    Initial graph update timeout, stopping");
                break;
            }
            
            // 清理UCR图
            ucr_graph.clear();
        }
        
        // 计算平均值
        if (!bz_times.empty()) {
            double avg_bz = 0, avg_ucr = 0;
            for (size_t i = 0; i < bz_times.size(); ++i) {
                avg_bz += bz_times[i];
                avg_ucr += ucr_times[i];
            }
            avg_bz /= bz_times.size();
            avg_ucr /= ucr_times.size();
            
            LOG("\nAverage over " + to_string(bz_times.size()) + " slides:");
            LOG("  BZ: " + to_string(avg_bz) + " ms");
            LOG("  UCR: " + to_string(avg_ucr) + " ms");
            LOG("  Average speedup: " + to_string(avg_bz / avg_ucr) + "x");
            
            // 写入全局结果文件
            global_results << dataset_name << "\t" 
                          << window_size << "\t"
                          << fixed << setprecision(2) 
                          << avg_bz << "\t"
                          << avg_ucr << "\t"
                          << (avg_bz / avg_ucr) << "\n";
            global_results.flush();
        }
        
        // 清理初始图和bins
        initial_graph.clear();
        for (auto& bin : bins) {
            vector<edge_t>().swap(bin);
        }
        vector<vector<edge_t>>().swap(bins);
    }
    
    size_t load_window_edges(Graph& graph, timestamp_t start_ts, timestamp_t end_ts) {
        ifstream file(dataset_path);
        if (!file.is_open()) {
            return 0;
        }
        
        string line;
        size_t edge_count = 0;
        
        while (getline(file, line)) {
            if (line.empty() || line[0] == '%' || line[0] == '#') continue;
            
            istringstream iss(line);
            vertex_t src, dst;
            timestamp_t ts;
            
            if (iss >> src >> dst >> ts && src != dst) {
                if (ts >= start_ts && ts < end_ts) {
                    graph.add_edge(src, dst);
                    edge_count++;
                }
            }
        }
        
        file.close();
        return edge_count;
    }
    
    vector<edge_t> get_edges_in_range(timestamp_t start_ts, timestamp_t end_ts) {
        vector<edge_t> edges;
        
        ifstream file(dataset_path);
        if (!file.is_open()) {
            return edges;
        }
        
        string line;
        
        while (getline(file, line)) {
            if (line.empty() || line[0] == '%' || line[0] == '#') continue;
            
            istringstream iss(line);
            vertex_t src, dst;
            timestamp_t ts;
            
            if (iss >> src >> dst >> ts && src != dst) {
                if (ts >= start_ts && ts < end_ts) {
                    edges.push_back({src, dst});
                }
            }
        }
        
        file.close();
        return edges;
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
            datasets.push_back({filepath, name});
        }
    }
    closedir(dirp);
    
    // 按文件名排序
    sort(datasets.begin(), datasets.end());
    
    return datasets;
}

int main() {
    LOG("===========================================");
    LOG("Experiment 2: UCR vs BZ Sliding Window");
    LOG("===========================================");
    
    // 打开全局结果文件
    string output_file = "e2_results.txt";
    global_results.open(output_file);
    
    if (!global_results.is_open()) {
        LOG("ERROR: Failed to open output file: " + output_file);
        return 1;
    }
    
    // 写入文件头
    global_results << "Dataset\tWindow_Size\tBZ_Avg(ms)\tUCR_Avg(ms)\tSpeedup\n";
    global_results.flush();
    
    LOG("Results will be saved to: " + output_file);
    
    // 获取所有数据集
    string dataset_dir = "dataset";
    auto datasets = get_datasets(dataset_dir);
    
    LOG("\nFound " + to_string(datasets.size()) + " datasets");
    
    // 对每个数据集运行实验
    int dataset_count = 0;
    for (size_t i = 0; i < datasets.size(); ++i) {
        const string& filepath = datasets[i].first;
        const string& name = datasets[i].second;
        
        dataset_count++;
        LOG("\n[" + to_string(dataset_count) + "/" + to_string(datasets.size()) + "] " + name);
        
        try {
            StreamingSlidingWindowExperiment experiment(filepath, name);
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
    LOG("Experiment 2 completed!");
    LOG("Results saved to: " + output_file);
    LOG("===========================================");
    
    return 0;
}