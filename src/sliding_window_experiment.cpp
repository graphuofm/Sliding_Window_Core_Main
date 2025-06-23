#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <memory>
#include <dirent.h>
#include <cstring>

// 定义顶点和边的类型
using vertex_t = int;
using edge_t = std::pair<vertex_t, vertex_t>;
using timestamp_t = long long;

// 时序边结构
struct TemporalEdge {
   vertex_t src;
   vertex_t dst;
   timestamp_t timestamp;
   
   TemporalEdge(vertex_t s, vertex_t d, timestamp_t t) : src(s), dst(d), timestamp(t) {}
};

// 计时辅助类
class Timer {
private:
   std::chrono::high_resolution_clock::time_point start_time;
public:
   Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
   
   double elapsed_milliseconds() const {
       auto end_time = std::chrono::high_resolution_clock::now();
       return std::chrono::duration<double, std::milli>(end_time - start_time).count();
   }
   
   void reset() {
       start_time = std::chrono::high_resolution_clock::now();
   }
};

// 图结构
class Graph {
public:
   std::vector<std::vector<vertex_t>> adj;
   std::vector<int> core;
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
       ensure_vertex(std::max(u, v));
       
       if (std::find(adj[u].begin(), adj[u].end(), v) == adj[u].end()) {
           adj[u].push_back(v);
       }
       if (std::find(adj[v].begin(), adj[v].end(), u) == adj[v].end()) {
           adj[v].push_back(u);
       }
   }
   
   void remove_edge(vertex_t u, vertex_t v) {
       if (u >= num_vertices || v >= num_vertices) return;
       
       auto it_u = std::find(adj[u].begin(), adj[u].end(), v);
       if (it_u != adj[u].end()) {
           *it_u = adj[u].back();
           adj[u].pop_back();
       }
       
       auto it_v = std::find(adj[v].begin(), adj[v].end(), u);
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
       
       std::fill(core.begin(), core.end(), 0);
       
       // 计算度数
       std::vector<int> degree(num_vertices);
       int max_degree = 0;
       for (vertex_t v = 0; v < num_vertices; ++v) {
           degree[v] = adj[v].size();
           max_degree = std::max(max_degree, degree[v]);
       }
       
       // 按度数分桶
       std::vector<std::vector<vertex_t>> bins(max_degree + 1);
       for (vertex_t v = 0; v < num_vertices; ++v) {
           bins[degree[v]].push_back(v);
       }
       
       // 核心度算法
       std::vector<bool> processed(num_vertices, false);
       
       for (int d = 0; d <= max_degree; ++d) {
           for (vertex_t v : bins[d]) {
               if (processed[v]) continue;
               
               core[v] = d;
               processed[v] = true;
               
               for (vertex_t w : adj[v]) {
                   if (processed[w]) continue;
                   
                   if (degree[w] > d) {
                       auto& bin = bins[degree[w]];
                       auto it = std::find(bin.begin(), bin.end(), w);
                       if (it != bin.end()) {
                           std::swap(*it, bin.back());
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
};

// MCD/PCD算法实现
class MCDPCDCore {
private:
   Graph& G;
   std::vector<int> MCD;
   std::vector<int> PCD;
   
   bool qual(vertex_t v, vertex_t w) const {
       if (v >= G.num_vertices || w >= G.num_vertices) return false;
       int k = G.core[v];
       return G.core[w] > k || (G.core[w] == k && MCD[w] > k);
   }
   
public:
   MCDPCDCore(Graph& g) : G(g) {
       reset();
   }
   
   void reset() {
       size_t n = G.num_vertices;
       MCD.resize(n, 0);
       PCD.resize(n, 0);
       
       // 计算MCD
       for (vertex_t v = 0; v < n; ++v) {
           int count = 0;
           for (vertex_t w : G.adj[v]) {
               if (G.core[w] >= G.core[v]) {
                   ++count;
               }
           }
           MCD[v] = count;
       }
       
       // 计算PCD
       for (vertex_t v = 0; v < n; ++v) {
           int count = 0;
           for (vertex_t w : G.adj[v]) {
               if (qual(v, w)) {
                   ++count;
               }
           }
           PCD[v] = count;
       }
   }
   
   double process_edge_removal(vertex_t u, vertex_t v) {
       Timer timer;
       
       if (u >= G.num_vertices || v >= G.num_vertices) return 0.0;
       
       // 更新MCD
       if (G.core[v] >= G.core[u]) --MCD[u];
       if (G.core[u] >= G.core[v]) --MCD[v];
       
       G.remove_edge(u, v);
       
       // 处理核心度降级
       int ku = G.core[u];
       int kv = G.core[v];
       
       std::vector<vertex_t> to_process;
       if (ku > 0) to_process.push_back(u);
       if (kv > 0) to_process.push_back(v);
       
       for (vertex_t seed : to_process) {
           int k = G.core[seed];
           if (k == 0) continue;
           
           std::queue<vertex_t> q;
           std::unordered_set<vertex_t> visited;
           q.push(seed);
           visited.insert(seed);
           
           std::vector<vertex_t> component;
           while (!q.empty()) {
               vertex_t v = q.front();
               q.pop();
               
               if (G.core[v] == k) {
                   component.push_back(v);
                   for (vertex_t w : G.adj[v]) {
                       if (G.core[w] == k && !visited.count(w)) {
                           visited.insert(w);
                           q.push(w);
                       }
                   }
               }
           }
           
           // 计算有效度数
           std::unordered_map<vertex_t, int> eff_deg;
           std::queue<vertex_t> degrade_queue;
           
           for (vertex_t v : component) {
               int count = 0;
               for (vertex_t w : G.adj[v]) {
                   if (G.core[w] >= k) count++;
               }
               eff_deg[v] = count;
               if (count < k) {
                   degrade_queue.push(v);
               }
           }
           
           // 级联降级
           while (!degrade_queue.empty()) {
               vertex_t v = degrade_queue.front();
               degrade_queue.pop();
               
               if (G.core[v] != k) continue;
               
               G.core[v] = k - 1;
               
               for (vertex_t w : G.adj[v]) {
                   if (G.core[w] == k && eff_deg[w] >= k) {
                       eff_deg[w]--;
                       if (eff_deg[w] < k) {
                           degrade_queue.push(w);
                       }
                   }
               }
           }
       }
       
       // 更新MCD和PCD
       reset();
       
       return timer.elapsed_milliseconds();
   }
   
   double process_edge_insertion(vertex_t u, vertex_t v) {
       Timer timer;
       
       G.ensure_vertex(std::max(u, v));
       
       if (G.adj[u].size() == 0) G.core[u] = 1;
       if (G.adj[v].size() == 0) G.core[v] = 1;
       
       G.add_edge(u, v);
       
       // 更新MCD
       if (G.core[v] >= G.core[u]) ++MCD[u];
       if (G.core[u] >= G.core[v]) ++MCD[v];
       
       // 检查是否可以升级
       std::unordered_set<vertex_t> candidates;
       if (PCD[u] > G.core[u]) candidates.insert(u);
       if (PCD[v] > G.core[v]) candidates.insert(v);
       
       if (!candidates.empty()) {
           for (vertex_t root : candidates) {
               int k = G.core[root];
               
               // 找连通分量
               std::queue<vertex_t> q;
               std::unordered_set<vertex_t> visited;
               q.push(root);
               visited.insert(root);
               
               std::vector<vertex_t> component;
               while (!q.empty()) {
                   vertex_t v = q.front();
                   q.pop();
                   
                   if (G.core[v] == k) {
                       component.push_back(v);
                       for (vertex_t w : G.adj[v]) {
                           if (G.core[w] == k && !visited.count(w)) {
                               visited.insert(w);
                               q.push(w);
                           }
                       }
                   }
               }
               
               // 模拟升级
               std::unordered_map<vertex_t, int> cd;
               for (vertex_t v : component) {
                   cd[v] = PCD[v];
               }
               
               std::unordered_set<vertex_t> evicted;
               bool stable = false;
               
               while (!stable) {
                   stable = true;
                   std::vector<vertex_t> to_evict;
                   
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
                           if (component.end() != std::find(component.begin(), component.end(), w) && 
                               !evicted.count(w) && G.core[w] == k && qual(w, v)) {
                               cd[w]--;
                           }
                       }
                   }
               }
               
               // 升级非evicted顶点
               for (vertex_t v : component) {
                   if (!evicted.count(v)) {
                       G.core[v]++;
                   }
               }
           }
       }
       
       // 更新MCD和PCD
       reset();
       
       return timer.elapsed_milliseconds();
   }
};

// UCR算法实现
class UCRCore {
private:
   Graph& G;
   std::vector<int> r_values;
   std::vector<int> s_values;
   
   bool is_qualified(vertex_t v) const {
       int k = G.core[v];
       return r_values[v] + s_values[v] > k;
   }
   
   void update_s_value(vertex_t v) {
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
   UCRCore(Graph& g) : G(g) {
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
   
   double process_edge_removal(vertex_t u, vertex_t v) {
       Timer timer;
       
       if (u >= G.num_vertices || v >= G.num_vertices) return 0.0;
       
       int ku = G.core[u];
       int kv = G.core[v];
       
       // 更新r值
       if (ku < kv) r_values[u]--;
       else if (ku > kv) r_values[v]--;
       
       G.remove_edge(u, v);
       
       // 处理降级
       vertex_t root = (ku <= kv) ? u : v;
       int k = G.core[root];
       
       if (k == 0) return timer.elapsed_milliseconds();
       
       // 找受影响的子核
       std::queue<vertex_t> q;
       std::unordered_set<vertex_t> visited;
       q.push(root);
       visited.insert(root);
       
       std::vector<vertex_t> subcore;
       std::unordered_map<vertex_t, int> cd;
       
       while (!q.empty()) {
           vertex_t v = q.front();
           q.pop();
           
           if (G.core[v] == k) {
               subcore.push_back(v);
               
               // 计算核心度
               int count = 0;
               for (vertex_t w : G.adj[v]) {
                   if (G.core[w] >= k) count++;
               }
               cd[v] = count;
               
               // 添加同核心度的邻居
               for (vertex_t w : G.adj[v]) {
                   if (G.core[w] == k && !visited.count(w)) {
                       visited.insert(w);
                       q.push(w);
                   }
               }
           }
       }
       
       // 找需要降级的顶点
       std::queue<vertex_t> degrade_queue;
       std::unordered_set<vertex_t> degraded;
       
       for (vertex_t v : subcore) {
           if (cd[v] < k) {
               degrade_queue.push(v);
           }
       }
       
       // 级联降级
       while (!degrade_queue.empty()) {
           vertex_t v = degrade_queue.front();
           degrade_queue.pop();
           
           if (G.core[v] != k || degraded.count(v)) continue;
           
           G.core[v] = k - 1;
           degraded.insert(v);
           
           // 更新邻居
           for (vertex_t w : G.adj[v]) {
               if (G.core[w] == k && !degraded.count(w)) {
                   cd[w]--;
                   if (cd[w] < k) {
                       degrade_queue.push(w);
                   }
               }
               
               // 更新r值
               if (G.core[w] > k - 1) {
                   r_values[w]--;
               }
           }
       }
       
       // 更新s值
       std::unordered_set<vertex_t> update_set;
       for (vertex_t v : degraded) {
           update_set.insert(v);
           for (vertex_t w : G.adj[v]) {
               update_set.insert(w);
           }
       }
       
       for (vertex_t v : update_set) {
           update_s_value(v);
       }
       
       return timer.elapsed_milliseconds();
   }
   
   double process_edge_insertion(vertex_t u, vertex_t v) {
       Timer timer;
       
       G.ensure_vertex(std::max(u, v));
       
       if (G.adj[u].size() == 0) G.core[u] = 1;
       if (G.adj[v].size() == 0) G.core[v] = 1;
       
       G.add_edge(u, v);
       
       int ku = G.core[u];
       int kv = G.core[v];
       
       // 更新r值
       if (ku < kv) r_values[u]++;
       else if (ku > kv) r_values[v]++;
       
       // 更新受影响顶点的s值
       std::unordered_set<vertex_t> affected;
       affected.insert(u);
       affected.insert(v);
       
       for (vertex_t v : affected) {
           update_s_value(v);
           for (vertex_t w : G.adj[v]) {
               if (G.core[w] == G.core[v]) {
                   update_s_value(w);
               }
           }
       }
       
       // 检查是否可以升级
       vertex_t root = (ku <= kv) ? u : v;
       int k = G.core[root];
       
       if (!is_qualified(root)) {
           return timer.elapsed_milliseconds();
       }
       
       // 找候选集
       std::queue<vertex_t> q;
       std::unordered_set<vertex_t> visited;
       q.push(root);
       visited.insert(root);
       
       std::vector<vertex_t> candidates;
       std::unordered_map<vertex_t, int> cd;
       
       while (!q.empty()) {
           vertex_t v = q.front();
           q.pop();
           
           if (G.core[v] == k && is_qualified(v)) {
               candidates.push_back(v);
               cd[v] = r_values[v] + s_values[v];
               
               for (vertex_t w : G.adj[v]) {
                   if (G.core[w] == k && !visited.count(w)) {
                       visited.insert(w);
                       q.push(w);
                   }
               }
           }
       }
       
       // 模拟升级
       std::unordered_set<vertex_t> evicted;
       bool stable = false;
       
       while (!stable) {
           stable = true;
           std::vector<vertex_t> to_evict;
           
           for (vertex_t v : candidates) {
               if (evicted.count(v)) continue;
               if (cd[v] <= k) {
                   to_evict.push_back(v);
                   stable = false;
               }
           }
           
           for (vertex_t v : to_evict) {
               evicted.insert(v);
               
               for (vertex_t w : G.adj[v]) {
                   if (G.core[w] == k && !evicted.count(w) && is_qualified(v)) {
                       cd[w]--;
                   }
               }
           }
       }
       
       // 升级
       std::unordered_set<vertex_t> upgraded;
       for (vertex_t v : candidates) {
           if (!evicted.count(v)) {
               G.core[v]++;
               upgraded.insert(v);
           }
       }
       
       // 更新r值和s值
       for (vertex_t v : upgraded) {
           for (vertex_t w : G.adj[v]) {
               if (G.core[w] < G.core[v]) {
                   r_values[w]++;
               }
           }
       }
       
       std::unordered_set<vertex_t> update_set;
       for (vertex_t v : upgraded) {
           update_set.insert(v);
           for (vertex_t w : G.adj[v]) {
               update_set.insert(w);
               for (vertex_t z : G.adj[w]) {
                   if (G.core[z] == G.core[w]) {
                       update_set.insert(z);
                   }
               }
           }
       }
       
       for (vertex_t v : update_set) {
           update_s_value(v);
       }
       
       return timer.elapsed_milliseconds();
   }
};

// 主实验类
class SlidingWindowExperiment {
private:
   std::string dataset_path;
   std::string dataset_name;
   std::vector<TemporalEdge> edges;
   std::vector<std::vector<edge_t>> bins;
   vertex_t max_vertex_id = 0;
   
   struct Result {
       int window_size;
       int slide_step;
       double bz_time;
       double mcd_time;
       double ucr_time;
   };
   
   std::vector<Result> results;
   
public:
   SlidingWindowExperiment(const std::string& path, const std::string& name) 
       : dataset_path(path), dataset_name(name) {}
   
   void run() {
       std::cout << "Processing dataset: " << dataset_name << std::endl;
       
       // 加载数据
       load_dataset();
       
       // 创建bins
       create_bins(1000);
       
       // 对每个窗口大小运行实验
       std::vector<int> window_sizes = {10, 20, 50, 100, 200};
       for (int window_size : window_sizes) {
           if (window_size > bins.size()) continue;
           run_window_experiment(window_size);
       }
       
       // 保存结果
       save_results();
   }
   
private:
   void load_dataset() {
       Timer timer;
       std::ifstream file(dataset_path);
       if (!file.is_open()) {
           std::cerr << "Failed to open file: " << dataset_path << std::endl;
           return;
       }
       
       std::string line;
       edges.reserve(1000000);
       
       while (std::getline(file, line)) {
           if (line.empty() || line[0] == '%' || line[0] == '#') continue;
           
           std::istringstream iss(line);
           vertex_t src, dst;
           timestamp_t ts;
           
           if (iss >> src >> dst >> ts) {
               edges.emplace_back(src, dst, ts);
               max_vertex_id = std::max(max_vertex_id, std::max(src, dst));
           }
       }
       
       std::cout << "Loaded " << edges.size() << " edges in " 
                 << timer.elapsed_milliseconds() << " ms" << std::endl;
   }
   
   void create_bins(int num_bins) {
       if (edges.empty()) return;
       
       timestamp_t min_ts = edges[0].timestamp;
       timestamp_t max_ts = edges[0].timestamp;
       
       for (const auto& e : edges) {
           min_ts = std::min(min_ts, e.timestamp);
           max_ts = std::max(max_ts, e.timestamp);
       }
       
       double bin_span = static_cast<double>(max_ts - min_ts + 1) / num_bins;
       bins.resize(num_bins);
       
       for (const auto& e : edges) {
           int bin_idx = std::min(static_cast<int>((e.timestamp - min_ts) / bin_span), num_bins - 1);
           bins[bin_idx].push_back({e.src, e.dst});
       }
       
       std::cout << "Created " << num_bins << " bins" << std::endl;
   }
   
   void run_window_experiment(int window_size) {
       std::cout << "\nRunning experiment with window size: " << window_size << std::endl;
       
       int starting_bin = 400;  // 固定起始位置
       if (starting_bin + window_size > bins.size()) {
           starting_bin = bins.size() - window_size - 10;
       }
       if (starting_bin < 0) starting_bin = 0;
       
       // 构建初始窗口
       Graph initial_graph;
       for (int i = starting_bin; i < starting_bin + window_size; ++i) {
           for (const auto& edge : bins[i]) {
               initial_graph.add_edge(edge.first, edge.second);
           }
       }
       
       // 计算初始核心度
       initial_graph.compute_core_numbers_bz();
       
       // 滑动10次
       for (int slide = 0; slide < 10; ++slide) {
           int remove_bin = starting_bin + slide;
           int add_bin = starting_bin + window_size + slide;
           
           if (add_bin >= bins.size()) break;
           
           std::cout << "Slide " << (slide + 1) << "/10" << std::endl;
           
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
           
           // MCD/PCD
           Timer mcd_timer;
           Graph mcd_graph = initial_graph.copy();
           MCDPCDCore mcd_core(mcd_graph);
           
           // 删除旧边
           for (const auto& edge : bins[remove_bin]) {
               mcd_core.process_edge_removal(edge.first, edge.second);
           }
           // 添加新边
           for (const auto& edge : bins[add_bin]) {
               mcd_core.process_edge_insertion(edge.first, edge.second);
           }
           double mcd_time = mcd_timer.elapsed_milliseconds();
           
           // UCR
           Timer ucr_timer;
           Graph ucr_graph = initial_graph.copy();
           UCRCore ucr_core(ucr_graph);
           
           // 删除旧边
           for (const auto& edge : bins[remove_bin]) {
               ucr_core.process_edge_removal(edge.first, edge.second);
           }
           // 添加新边
           for (const auto& edge : bins[add_bin]) {
               ucr_core.process_edge_insertion(edge.first, edge.second);
           }
           double ucr_time = ucr_timer.elapsed_milliseconds();
           
           // 保存结果
           results.push_back({window_size, slide + 1, bz_time, mcd_time, ucr_time});
           
           // 更新初始图为下一次滑动准备
           for (const auto& edge : bins[remove_bin]) {
               initial_graph.remove_edge(edge.first, edge.second);
           }
           for (const auto& edge : bins[add_bin]) {
               initial_graph.add_edge(edge.first, edge.second);
           }
           initial_graph.compute_core_numbers_bz();
       }
   }
   
   void save_results() {
       std::string output_file = "/home/jding/" + dataset_name + "_sliding_window_results.txt";
       std::ofstream out(output_file);
       
       out << "Dataset: " << dataset_name << "\n";
       out << "Window_Size\tSlide_Step\tBZ_Time(ms)\tMCD_Time(ms)\tUCR_Time(ms)\n";
       
       for (const auto& r : results) {
           out << r.window_size << "\t" 
               << r.slide_step << "\t"
               << std::fixed << std::setprecision(2) 
               << r.bz_time << "\t"
               << r.mcd_time << "\t"
               << r.ucr_time << "\n";
       }
       
       out.close();
       std::cout << "Results saved to: " << output_file << std::endl;
   }
};

// 获取数据集文件列表
std::vector<std::pair<std::string, std::string>> get_datasets(const std::string& dir) {
   std::vector<std::pair<std::string, std::string>> datasets;
   
   DIR* dirp = opendir(dir.c_str());
   if (dirp == nullptr) {
       std::cerr << "Failed to open directory: " << dir << std::endl;
       return datasets;
   }
   
   struct dirent* dp;
   while ((dp = readdir(dirp)) != nullptr) {
       std::string filename(dp->d_name);
       if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".txt") {
           std::string filepath = dir + "/" + filename;
           std::string name = filename.substr(0, filename.size() - 4);
           datasets.push_back({filepath, name});
       }
   }
   closedir(dirp);
   
   return datasets;
}

int main() {
   // 获取所有数据集
   std::string dataset_dir = "/home/jding/dataset";
   auto datasets = get_datasets(dataset_dir);
   
   std::cout << "Found " << datasets.size() << " datasets" << std::endl;
   
   // 对每个数据集运行实验
   for (const auto& [filepath, name] : datasets) {
       std::cout << "\n========================================" << std::endl;
       std::cout << "Processing: " << name << std::endl;
       std::cout << "========================================" << std::endl;
       
       try {
           SlidingWindowExperiment experiment(filepath, name);
           experiment.run();
       } catch (const std::exception& e) {
           std::cerr << "Error processing " << name << ": " << e.what() << std::endl;
       }
   }
   
   std::cout << "\nAll experiments completed!" << std::endl;
   return 0;
}