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
        ensure_vertex(max(u, v));
        
        // 避免自环和重复边
        if (u == v) return;
        
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
    
    size_t degree(vertex_t v) const {
        if (v >= num_vertices) return 0;
        return adj[v].size();
    }
    
    // BZ算法计算核心度
    void compute_core_numbers_bz() {
        if (num_vertices == 0) return;
        
        fill(core.begin(), core.end(), 0);
        
        vector<int> degree(num_vertices);
        int max_degree = 0;
        for (vertex_t v = 0; v < num_vertices; ++v) {
            degree[v] = adj[v].size();
            max_degree = max(max_degree, degree[v]);
        }
        
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
    
    size_t num_edges() const {
        size_t edges = 0;
        for (const auto& neighbors : adj) {
            edges += neighbors.size();
        }
        return edges / 2;
    }
};

// UCR算法实现（批处理优化版本）
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
        
        s_values[v] = s_count;
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
    
    // 批量处理边的删除和插入
    double process_batch(const vector<edge_t>& remove_edges, 
                        const vector<edge_t>& add_edges) {
        Timer timer;
        
        // Phase 1: 批量删除边
        unordered_map<int, unordered_set<vertex_t>> k_groups_remove;
        
        // 更新r值并删除边
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
        
        // 按核心度层次处理删除
        vector<int> k_values;
        for (const auto& pair : k_groups_remove) {
            k_values.push_back(pair.first);
        }
        sort(k_values.begin(), k_values.end());
        
        for (int k : k_values) {
            process_core_level_removal(k, k_groups_remove[k]);
        }
        
        // Phase 2: 批量插入边
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
        
        // 添加边并收集受影响顶点
        unordered_set<vertex_t> affected_vertices;
        
        for (const auto& edge : add_edges) {
            vertex_t u = edge.first;
            vertex_t v = edge.second;
            
            G.ensure_vertex(max(u, v));
            
            if (u == v) continue;  // 跳过自环
            
            if (G.adj[u].size() == 0) G.core[u] = 1;
            if (G.adj[v].size() == 0) G.core[v] = 1;
            
            G.add_edge(u, v);
            
            int ku = G.core[u];
            int kv = G.core[v];
            
            if (ku < kv) r_values[u]++;
            else if (ku > kv) r_values[v]++;
            
            affected_vertices.insert(u);
            affected_vertices.insert(v);
        }
        
        // 批量更新s值
        unordered_set<vertex_t> s_update_set;
        for (vertex_t v : affected_vertices) {
            s_update_set.insert(v);
            for (vertex_t w : G.adj[v]) {
                if (G.core[w] == G.core[v]) {
                    s_update_set.insert(w);
                }
            }
        }
        
        for (vertex_t v : s_update_set) {
            update_s_value(v);
        }
        
        // 找所有可能升级的候选
        unordered_set<vertex_t> all_candidates;
        for (vertex_t v : affected_vertices) {
            if (is_qualified(v)) {
                all_candidates.insert(v);
            }
        }
        
        // 批量处理升级
        if (!all_candidates.empty()) {
            vector<vector<vertex_t>> components;
            identify_components(all_candidates, components);
            
            for (const auto& component : components) {
                process_component_upgrade(component);
            }
        }
        
        return timer.elapsed_milliseconds();
    }
    
private:
    void process_core_level_removal(int k, const unordered_set<vertex_t>& seeds) {
        unordered_set<vertex_t> visited;
        
        for (vertex_t seed : seeds) {
            if (visited.count(seed) || G.core[seed] != k) continue;
            
            // 找连通分量
            queue<vertex_t> q;
            q.push(seed);
            visited.insert(seed);
            
            vector<vertex_t> component;
            unordered_map<vertex_t, int> eff_deg;
            
            while (!q.empty()) {
                vertex_t v = q.front();
                q.pop();
                
                if (G.core[v] == k) {
                    component.push_back(v);
                    
                    int count = 0;
                    for (vertex_t w : G.adj[v]) {
                        if (G.core[w] >= k) count++;
                    }
                    eff_deg[v] = count;
                    
                    for (vertex_t w : G.adj[v]) {
                        if (G.core[w] == k && !visited.count(w)) {
                            visited.insert(w);
                            q.push(w);
                        }
                    }
                }
            }
            
            // 批量级联降级
            queue<vertex_t> degrade_queue;
            unordered_set<vertex_t> degraded;
            
            for (vertex_t v : component) {
                if (eff_deg[v] < k) {
                    degrade_queue.push(v);
                }
            }
            
            while (!degrade_queue.empty()) {
                vertex_t v = degrade_queue.front();
                degrade_queue.pop();
                
                if (G.core[v] != k || degraded.count(v)) continue;
                
                G.core[v] = k - 1;
                degraded.insert(v);
                
                for (vertex_t w : G.adj[v]) {
                    if (G.core[w] == k && !degraded.count(w)) {
                        eff_deg[w]--;
                        if (eff_deg[w] < k) {
                            degrade_queue.push(w);
                        }
                    }
                    
                    if (w < r_values.size() && G.core[w] > k - 1) {
                        r_values[w]--;
                    }
                }
            }
            
            // 批量更新s值
            unordered_set<vertex_t> update_set;
            for (vertex_t v : degraded) {
                update_set.insert(v);
                for (vertex_t w : G.adj[v]) {
                    update_set.insert(w);
                }
            }
            
            for (vertex_t v : update_set) {
                update_s_value(v);
            }
        }
    }
    
    void identify_components(const unordered_set<vertex_t>& candidates,
                            vector<vector<vertex_t>>& components) {
        unordered_set<vertex_t> visited;
        
        for (vertex_t v : candidates) {
            if (visited.count(v)) continue;
            
            vector<vertex_t> component;
            queue<vertex_t> q;
            q.push(v);
            visited.insert(v);
            
            int k = G.core[v];
            
            while (!q.empty()) {
                vertex_t u = q.front();
                q.pop();
                
                if (G.core[u] == k) {
                    component.push_back(u);
                    
                    for (vertex_t w : G.adj[u]) {
                        if (G.core[w] == k && !visited.count(w) && candidates.count(w)) {
                            visited.insert(w);
                            q.push(w);
                        }
                    }
                }
            }
            
            if (!component.empty()) {
                components.push_back(move(component));
            }
        }
    }
    
    void process_component_upgrade(const vector<vertex_t>& component) {
        if (component.empty()) return;
        
        int k = G.core[component[0]];
        
        unordered_map<vertex_t, int> cd;
        for (vertex_t v : component) {
            cd[v] = r_values[v] + s_values[v];
        }
        
        unordered_set<vertex_t> evicted;
        bool stable = false;
        
        while (!stable) {
            stable = true;
            vector<vertex_t> to_evict;
            
            for (vertex_t v : component) {
                if (evicted.count(v)) continue;
                if (cd[v] <= k) {
                    to_evict.push_back(v);
                    stable = false;
                }
            }
            
            for (vertex_t v : to_evict) {
                evicted.insert(v);
                
                for (vertex_t w : G.adj[v]) {
                    if (G.core[w] == k && !evicted.count(w) && cd.count(w)) {
                        cd[w]--;
                    }
                }
            }
        }
        
        // 批量升级
        for (vertex_t v : component) {
            if (!evicted.count(v)) {
                G.core[v]++;
                
                for (vertex_t w : G.adj[v]) {
                    if (w < r_values.size() && G.core[w] < G.core[v]) {
                        r_values[w]++;
                    }
                }
            }
        }
        
        // 批量更新s值
        unordered_set<vertex_t> update_set;
        for (vertex_t v : component) {
            if (!evicted.count(v)) {
                update_set.insert(v);
                for (vertex_t w : G.adj[v]) {
                    update_set.insert(w);
                }
            }
        }
        
        for (vertex_t v : update_set) {
            update_s_value(v);
        }
    }
};

// 滑动窗口实验类
class SlidingWindowExperiment {
private:
    string dataset_path;
    string dataset_name;
    vector<TemporalEdge> edges;
    vector<vector<edge_t>> bins;
    
public:
    SlidingWindowExperiment(const string& path, const string& name) 
        : dataset_path(path), dataset_name(name) {}
    
    void run() {
        LOG("=====================================");
        LOG("Dataset: " + dataset_name);
        LOG("=====================================");
        
        // 加载数据
        if (!load_dataset()) {
            LOG("ERROR: Failed to load dataset " + dataset_name);
            return;
        }
        
        // 创建1000个bins
        create_bins(1000);
        
        // 对每个窗口大小运行实验
        vector<int> window_sizes = {10, 20, 50, 100};
        for (int window_size : window_sizes) {
            if (window_size > bins.size()) {
                LOG("Skipping window size " + to_string(window_size) + " (larger than number of bins)");
                continue;
            }
            run_window_experiment(window_size);
        }
    }
    
private:
    bool load_dataset() {
        Timer timer;
        LOG("Loading dataset from: " + dataset_path);
        
        ifstream file(dataset_path);
        if (!file.is_open()) {
            return false;
        }
        
        string line;
        edges.clear();
        size_t line_count = 0;
        
        while (getline(file, line)) {
            line_count++;
            if (line_count % 1000000 == 0) {
                LOG("  Loaded " + to_string(line_count) + " lines...");
            }
            
            if (line.empty() || line[0] == '%' || line[0] == '#') continue;
            
            istringstream iss(line);
            vertex_t src, dst;
            timestamp_t ts;
            
            if (iss >> src >> dst >> ts) {
                edges.emplace_back(src, dst, ts);
            }
        }
        
        file.close();
        
        LOG("  Total edges loaded: " + to_string(edges.size()));
        LOG("  Load time: " + to_string(timer.elapsed_milliseconds()) + " ms");
        
        return true;
    }
    
    void create_bins(int num_bins) {
        if (edges.empty()) {
            LOG("ERROR: No edges to bin");
            return;
        }
        
        LOG("Creating " + to_string(num_bins) + " bins...");
        
        // 按时间戳排序
        sort(edges.begin(), edges.end(), 
             [](const TemporalEdge& a, const TemporalEdge& b) {
                 return a.timestamp < b.timestamp;
             });
        
        timestamp_t min_ts = edges.front().timestamp;
        timestamp_t max_ts = edges.back().timestamp;
        
        double bin_span = static_cast<double>(max_ts - min_ts + 1) / num_bins;
        bins.clear();
        bins.resize(num_bins);
        
        for (const auto& e : edges) {
            int bin_idx = min(static_cast<int>((e.timestamp - min_ts) / bin_span), num_bins - 1);
            bins[bin_idx].push_back({e.src, e.dst});
        }
        
        LOG("  Binning complete");
    }
    
    void run_window_experiment(int window_size) {
        LOG("\n--- Window size: " + to_string(window_size) + " bins ---");
        
        // 选择起始位置
        int starting_bin = 400;
        if (starting_bin + window_size > bins.size()) {
            starting_bin = bins.size() - window_size - 10;
        }
        if (starting_bin < 0) starting_bin = 0;
        
        // 构建初始窗口
        LOG("Building initial window...");
        Graph initial_graph;
        size_t total_edges = 0;
        for (int i = starting_bin; i < starting_bin + window_size; ++i) {
            total_edges += bins[i].size();
            for (const auto& edge : bins[i]) {
                initial_graph.add_edge(edge.first, edge.second);
            }
        }
        LOG("  Initial edges: " + to_string(total_edges));
        
        // 计算初始核心度
        Timer init_timer;
        initial_graph.compute_core_numbers_bz();
        LOG("  Initial core computation: " + to_string(init_timer.elapsed_milliseconds()) + " ms");
        
        // UCR和BZ的时间记录
        vector<double> bz_times, ucr_times;
        
        // 滑动3次
        for (int slide = 0; slide < 3; ++slide) {
            int remove_bin = starting_bin + slide;
            int add_bin = starting_bin + window_size + slide;
            
            if (add_bin >= bins.size()) {
                LOG("Reached end of bins at slide " + to_string(slide + 1));
                break;
            }
            
            LOG("\nSlide " + to_string(slide + 1) + ":");
            LOG("  Remove: " + to_string(bins[remove_bin].size()) + " edges");
            LOG("  Add: " + to_string(bins[add_bin].size()) + " edges");
            
            // BZ重计算
            Timer bz_timer;
            Graph bz_graph;
            for (int i = remove_bin + 1; i <= add_bin; ++i) {
                for (const auto& edge : bins[i]) {
                    bz_graph.add_edge(edge.first, edge.second);
                }
            }
            bz_graph.compute_core_numbers_bz();
            double bz_time = bz_timer.elapsed_milliseconds();
            bz_times.push_back(bz_time);
            LOG("    BZ time: " + to_string(bz_time) + " ms");
            
            // UCR批处理
            Graph ucr_graph = initial_graph.copy();
            UCRBatch ucr_core(ucr_graph);
            
            Timer ucr_timer;
            double ucr_time = ucr_core.process_batch(bins[remove_bin], bins[add_bin]);
            ucr_times.push_back(ucr_time);
            LOG("    UCR time: " + to_string(ucr_time) + " ms");
            LOG("    Speedup: " + to_string(bz_time / ucr_time) + "x");
            
            // 更新初始图为下一次滑动准备
            for (const auto& edge : bins[remove_bin]) {
                initial_graph.remove_edge(edge.first, edge.second);
            }
            for (const auto& edge : bins[add_bin]) {
                initial_graph.add_edge(edge.first, edge.second);
            }
            initial_graph.compute_core_numbers_bz();
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
            
            // 检查文件大小，跳过太大的文件
            struct stat st;
            if (stat(filepath.c_str(), &st) == 0) {
                // 跳过大于2GB的文件，避免处理时间过长
                if (st.st_size > 2LL * 1024 * 1024 * 1024) {
                    LOG("Skipping large file: " + name + " (size: " + 
                        to_string(st.st_size / (1024*1024)) + " MB)");
                    continue;
                }
            }
            
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
    string output_file = "/home/jding/e2_ucr_vs_bz_results.txt";
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
    string dataset_dir = "/home/jding/dataset";
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
            SlidingWindowExperiment experiment(filepath, name);
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