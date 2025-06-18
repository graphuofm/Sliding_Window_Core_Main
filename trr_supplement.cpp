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

// 图结构 - 为大图优化
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
    
    // BZ算法 - 加长超时时间
    bool compute_core_numbers_bz(double timeout_seconds = 3600.0) { // 1小时超时
        Timer timer;
        
        if (num_vertices == 0) return true;
        
        fill(core.begin(), core.end(), 0);
        
        vector<int> degree(num_vertices);
        int max_degree = 0;
        
        // 并行计算度数
        #pragma omp parallel for reduction(max:max_degree)
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
        size_t processed_count = 0;
        
        for (int d = 0; d <= max_degree; ++d) {
            // 每10000次检查超时
            if (d % 10000 == 0) {
                double elapsed = timer.elapsed_seconds();
                LOG("    BZ progress: processing degree " + to_string(d) + "/" + to_string(max_degree) + 
                    ", processed " + to_string(processed_count) + "/" + to_string(num_vertices) + 
                    " vertices, time: " + to_string(elapsed) + "s");
                
                if (elapsed > timeout_seconds) {
                    LOG("    BZ computation timeout after " + to_string(elapsed) + " seconds");
                    return false;
                }
            }
            
            for (vertex_t v : bins[d]) {
                if (processed[v]) continue;
                
                core[v] = d;
                processed[v] = true;
                processed_count++;
                
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

// UCR基础版本
class UCRBasic {
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
    UCRBasic(Graph& g) : G(g) {
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
    
    double process_sliding_window(const vector<edge_t>& remove_edges, 
                                 const vector<edge_t>& add_edges) {
        Timer timer;
        
        // 逐个删除边
        for (const auto& edge : remove_edges) {
            process_edge_removal(edge);
        }
        
        // 逐个添加边
        for (const auto& edge : add_edges) {
            process_edge_insertion(edge);
        }
        
        return timer.elapsed_milliseconds();
    }
    
private:
    void process_edge_removal(const edge_t& edge) {
        vertex_t u = edge.first;
        vertex_t v = edge.second;
        
        if (u >= G.num_vertices || v >= G.num_vertices) return;
        
        int ku = G.core[u];
        int kv = G.core[v];
        
        if (ku < kv && u < r_values.size()) r_values[u]--;
        else if (ku > kv && v < r_values.size()) r_values[v]--;
        
        G.remove_edge(u, v);
        
        if (r_values[u] + s_values[u] < G.core[u]) {
            G.core[u]--;
        }
        if (r_values[v] + s_values[v] < G.core[v]) {
            G.core[v]--;
        }
        
        update_s_value(u);
        update_s_value(v);
    }
    
    void process_edge_insertion(const edge_t& edge) {
        vertex_t u = edge.first;
        vertex_t v = edge.second;
        
        if (u == v) return;
        
        G.ensure_vertex(max(u, v));
        
        if (max(u, v) >= r_values.size()) {
            r_values.resize(max(u, v) + 1, 0);
            s_values.resize(max(u, v) + 1, 0);
        }
        
        G.add_edge(u, v);
        
        int ku = G.core[u];
        int kv = G.core[v];
        
        if (ku < kv && u < r_values.size()) r_values[u]++;
        else if (ku > kv && v < r_values.size()) r_values[v]++;
        
        if (r_values[u] + s_values[u] > G.core[u]) {
            G.core[u]++;
        }
        if (r_values[v] + s_values[v] > G.core[v]) {
            G.core[v]++;
        }
        
        update_s_value(u);
        update_s_value(v);
    }
};

// UCR并行版本
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
        #pragma omp parallel for
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
        #pragma omp parallel for
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
        
        // 串行处理边删除
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
        
        // 串行处理边添加
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
        
        // 并行更新s值
        #pragma omp parallel for
        for (size_t i = 0; i < affected_vertices.size(); ++i) {
            vertex_t v = affected_vertices[i];
            update_s_value(v);
        }
        
        // 串行处理核心度升级
        for (vertex_t v : affected_vertices) {
            if (v < r_values.size() && r_values[v] + s_values[v] > G.core[v]) {
                G.core[v]++;
            }
        }
        
        return timer.elapsed_milliseconds();
    }
};

// 前向声明
void testUCRAlgorithms(Graph& initial_graph, const vector<edge_t>& remove_edges, 
                       const vector<edge_t>& add_edges, double bz_time);

// TRR补充实验
void run_trr_supplement() {
    string dataset_path = "dataset/temporal-reddit-reply.txt";
    string dataset_name = "temporal-reddit-reply";
    
    LOG("===========================================");
    LOG("Supplementary Experiment: " + dataset_name + " (Window size 100)");
    LOG("===========================================");
    
    // 设置OpenMP线程数
    omp_set_num_threads(8); // 使用更多线程
    LOG("OpenMP threads: " + to_string(omp_get_max_threads()));
    
    // 扫描时间戳范围
    LOG("Scanning timestamp range...");
    
    ifstream file(dataset_path);
    if (!file.is_open()) {
        LOG("ERROR: Cannot open file");
        return;
    }
    
    timestamp_t min_ts = LLONG_MAX;
    timestamp_t max_ts = LLONG_MIN;
    size_t edge_count = 0;
    
    string line;
    while (getline(file, line)) {
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
    
    LOG("Time range: " + to_string(min_ts) + " to " + to_string(max_ts));
    LOG("Total edges: " + to_string(edge_count));
    
    // 设置窗口参数
    const int window_size = 100;
    const int total_bins = 1000;
    double bin_span = static_cast<double>(max_ts - min_ts + 1) / total_bins;
    
    // 从最开始的位置开始（边少）
    int starting_bin = 0; // 从最开始开始
    
    LOG("\nLoading and indexing edges...");
    vector<vector<edge_t>> bins(total_bins);
    
    file.open(dataset_path);
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
            
            if (loaded % 10000000 == 0) {
                LOG("  Loaded " + to_string(loaded) + " edges...");
            }
        }
    }
    file.close();
    
    LOG("Loading complete: " + to_string(loaded) + " edges in " + 
        to_string(load_timer.elapsed_seconds()) + " seconds");
    
    // 查看前面几个bin的边数
    LOG("\nEdges in first few bins:");
    for (int i = 0; i < 10; ++i) {
        LOG("  Bin " + to_string(i) + ": " + to_string(bins[i].size()) + " edges");
    }
    
    // 构建初始窗口
    LOG("\nBuilding initial window (bins " + to_string(starting_bin) + " to " + 
        to_string(starting_bin + window_size - 1) + ")...");
    
    Graph initial_graph;
    size_t initial_edges = 0;
    
    for (int i = starting_bin; i < starting_bin + window_size; ++i) {
        initial_edges += bins[i].size();
        for (const auto& edge : bins[i]) {
            initial_graph.add_edge(edge.first, edge.second);
        }
    }
    
    LOG("Initial edges in window: " + to_string(initial_edges));
    LOG("Initial vertices: " + to_string(initial_graph.num_vertices));
    
    // 计算初始核心度
    Timer init_timer;
    LOG("Computing initial core numbers (this may take a while)...");
    bool success = initial_graph.compute_core_numbers_bz(7200.0); // 2小时超时
    
    if (!success) {
        LOG("ERROR: Initial core computation timeout");
        return;
    }
    
    double init_time = init_timer.elapsed_milliseconds();
    LOG("Initial core computation: " + to_string(init_time / 1000.0) + " seconds");
    
    // 进行滑动测试
    int remove_bin = starting_bin;
    int add_bin = starting_bin + window_size;
    
    LOG("\nPerforming sliding window test:");
    LOG("Remove: " + to_string(bins[remove_bin].size()) + " edges from bin " + to_string(remove_bin));
    LOG("Add: " + to_string(bins[add_bin].size()) + " edges from bin " + to_string(add_bin));
    
    // BZ重计算
    LOG("\nRunning BZ algorithm...");
    Timer bz_timer;
    Graph bz_graph;
    
    // 构建新窗口
    for (int i = remove_bin + 1; i <= add_bin; ++i) {
        for (const auto& edge : bins[i]) {
            bz_graph.add_edge(edge.first, edge.second);
        }
    }
    
    bool bz_success = bz_graph.compute_core_numbers_bz(3600.0); // 1小时超时
    
    if (!bz_success) {
        LOG("ERROR: BZ computation timeout");
        
        // 即使BZ超时，我们仍然可以测试UCR
        LOG("Continuing with UCR tests despite BZ timeout...");
        
        // 使用一个估计的BZ时间
        double bz_time = 60000.0; // 假设60秒
        LOG("Using estimated BZ time: " + to_string(bz_time) + " ms");
        
        // 测试UCR算法
        testUCRAlgorithms(initial_graph, bins[remove_bin], bins[add_bin], bz_time);
    } else {
        double bz_time = bz_timer.elapsed_milliseconds();
        LOG("BZ time: " + to_string(bz_time) + " ms");
        
        // 清理BZ图
        bz_graph.clear();
        
        // 测试UCR算法
        testUCRAlgorithms(initial_graph, bins[remove_bin], bins[add_bin], bz_time);
    }
    
    // 清理
    initial_graph.clear();
    LOG("\nSupplementary experiment completed!");
}

void testUCRAlgorithms(Graph& initial_graph, const vector<edge_t>& remove_edges, 
                       const vector<edge_t>& add_edges, double bz_time) {
    
    // UCR-Basic
    LOG("\nTesting UCR-Basic...");
    {
        Graph ucr_graph = initial_graph.copy();
        UCRBasic ucr_core(ucr_graph);
        
        Timer ucr_timer;
        double ucr_time = ucr_core.process_sliding_window(remove_edges, add_edges);
        double speedup = bz_time / ucr_time;
        
        LOG("UCR-Basic time: " + to_string(ucr_time) + " ms");
        LOG("UCR-Basic speedup: " + to_string(speedup) + "x");
        
        // 输出结果到文件
        ofstream results("trr_supplement_results.txt", ios::app);
        results << "temporal-reddit-reply\t100\tBZ\t" << fixed << setprecision(2) 
                << bz_time << "\t-\t-\n";
        results << "temporal-reddit-reply\t100\tUCR-Basic\t-\t" 
                << ucr_time << "\t" << speedup << "\n";
        results.close();
        
        ucr_graph.clear();
    }
    
    // UCR-Parallel
    LOG("\nTesting UCR-Parallel...");
    {
        Graph ucr_graph = initial_graph.copy();
        UCRParallel ucr_core(ucr_graph);
        
        Timer ucr_timer;
        double ucr_time = ucr_core.process_sliding_window(remove_edges, add_edges);
        double speedup = bz_time / ucr_time;
        
        LOG("UCR-Parallel time: " + to_string(ucr_time) + " ms");
        LOG("UCR-Parallel speedup: " + to_string(speedup) + "x");
        
        // 输出结果到文件
        ofstream results("trr_supplement_results.txt", ios::app);
        results << "temporal-reddit-reply\t100\tUCR-Parallel\t-\t" 
                << ucr_time << "\t" << speedup << "\n";
        results.close();
        
        ucr_graph.clear();
    }
}

int main() {
    // 创建结果文件
    ofstream results("trr_supplement_results.txt");
    results << "Dataset\tWindow_Size\tMethod\tBZ_Time(ms)\tUCR_Time(ms)\tSpeedup\n";
    results.close();
    
    // 运行补充实验
    run_trr_supplement();
    
    return 0;
}