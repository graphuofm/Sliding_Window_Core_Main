#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <sstream>
#include <unordered_map>
#include <iomanip>

using namespace std;
using vertex_t = int;

// 高精度计时器类
class Timer {
private:
    chrono::high_resolution_clock::time_point start_time;
public:
    Timer() : start_time(chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_time = chrono::high_resolution_clock::now();
    }
    
    double elapsed_seconds() const {
        auto end_time = chrono::high_resolution_clock::now();
        return chrono::duration<double>(end_time - start_time).count();
    }
};

// 图结构
class Graph {
public:
    vector<vector<vertex_t>> adj;
    size_t num_vertices;
    size_t num_edges;
    size_t self_loops_skipped;
    
    Graph() : num_vertices(0), num_edges(0), self_loops_skipped(0) {}
    
    void ensure_vertex(vertex_t v) {
        if (v >= adj.size()) {
            adj.resize(v + 1);
            num_vertices = v + 1;
        }
    }
    
    void add_edge(vertex_t u, vertex_t v) {
        // 跳过自环边
        if (u == v) {
            self_loops_skipped++;
            return;
        }
        
        ensure_vertex(max(u, v));
        
        // 检查是否已存在（避免重复边）
        if (find(adj[u].begin(), adj[u].end(), v) == adj[u].end()) {
            adj[u].push_back(v);
            adj[v].push_back(u);
            num_edges++;
        }
    }
    
    size_t degree(vertex_t v) const {
        if (v >= adj.size()) return 0;
        return adj[v].size();
    }
};

// BZ算法实现
class BZAlgorithm {
private:
    const Graph& G;
    vector<int> core;
    
public:
    BZAlgorithm(const Graph& g) : G(g), core(g.num_vertices, 0) {}
    
    void compute() {
        Timer timer;
        cout << "\nComputing k-core decomposition with BZ algorithm..." << endl;
        
        size_t n = G.num_vertices;
        if (n == 0) return;
        
        // 计算所有顶点的度数
        vector<int> degree(n);
        int max_degree = 0;
        
        for (vertex_t v = 0; v < n; ++v) {
            degree[v] = G.degree(v);
            max_degree = max(max_degree, degree[v]);
        }
        
        // 创建度数桶
        vector<vector<vertex_t>> bins(max_degree + 1);
        vector<int> pos(n);  // 记录顶点在桶中的位置
        
        for (vertex_t v = 0; v < n; ++v) {
            bins[degree[v]].push_back(v);
            pos[v] = bins[degree[v]].size() - 1;
        }
        
        // BZ算法主循环
        vector<bool> processed(n, false);
        
        for (int d = 0; d <= max_degree; ++d) {
            while (!bins[d].empty()) {
                vertex_t v = bins[d].back();
                bins[d].pop_back();
                
                if (processed[v]) continue;
                
                core[v] = d;
                processed[v] = true;
                
                // 更新邻居的度数
                for (vertex_t w : G.adj[v]) {
                    if (processed[w]) continue;
                    
                    if (degree[w] > d) {
                        // 从当前桶中移除w
                        int old_degree = degree[w];
                        int w_pos = pos[w];
                        
                        // 将最后一个元素移到w的位置
                        if (w_pos < bins[old_degree].size()) {
                            vertex_t last = bins[old_degree].back();
                            bins[old_degree][w_pos] = last;
                            pos[last] = w_pos;
                        }
                        bins[old_degree].pop_back();
                        
                        // 减少度数并加入新桶
                        degree[w]--;
                        bins[degree[w]].push_back(w);
                        pos[w] = bins[degree[w]].size() - 1;
                    }
                }
            }
        }
        
        double elapsed = timer.elapsed_seconds();
        cout << "K-core computation completed in " << fixed << setprecision(3) 
             << elapsed << " seconds" << endl;
    }
    
    const vector<int>& get_core_numbers() const {
        return core;
    }
    
    void analyze_distribution() const {
        cout << "\nCore number distribution:" << endl;
        
        // 统计核心数分布
        unordered_map<int, int> core_dist;
        int max_core = 0;
        int vertices_with_core = 0;
        
        for (vertex_t v = 0; v < core.size(); ++v) {
            if (G.degree(v) > 0) {  // 只统计有边的顶点
                core_dist[core[v]]++;
                max_core = max(max_core, core[v]);
                vertices_with_core++;
            }
        }
        
        cout << "  Max core number: " << max_core << endl;
        cout << "  Vertices with edges: " << vertices_with_core << endl;
        
        // 排序并显示分布
        vector<pair<int, int>> sorted_dist(core_dist.begin(), core_dist.end());
        sort(sorted_dist.begin(), sorted_dist.end());
        
        cout << "  Core distribution (showing first 10 and last 5):" << endl;
        
        for (int i = 0; i < min(10, (int)sorted_dist.size()); ++i) {
            cout << "    " << sorted_dist[i].first << "-core: " 
                 << sorted_dist[i].second << " vertices" << endl;
        }
        
        if (sorted_dist.size() > 15) {
            cout << "    ..." << endl;
            for (int i = max(10, (int)sorted_dist.size() - 5); i < sorted_dist.size(); ++i) {
                cout << "    " << sorted_dist[i].first << "-core: " 
                     << sorted_dist[i].second << " vertices" << endl;
            }
        } else if (sorted_dist.size() > 10) {
            for (int i = 10; i < sorted_dist.size(); ++i) {
                cout << "    " << sorted_dist[i].first << "-core: " 
                     << sorted_dist[i].second << " vertices" << endl;
            }
        }
    }
    
    void save_results(const string& filename) const {
        cout << "\nSaving results to " << filename << "..." << endl;
        
        ofstream out(filename);
        if (!out.is_open()) {
            cerr << "Error: Cannot open output file!" << endl;
            return;
        }
        
        out << "# C++ BZ algorithm k-core results" << endl;
        out << "# Format: vertex_id core_number" << endl;
        out << "# Total vertices: " << core.size() << endl;
        
        // 只保存有边的顶点的核心数
        int saved_count = 0;
        for (vertex_t v = 0; v < core.size(); ++v) {
            if (G.degree(v) > 0) {
                out << v << " " << core[v] << endl;
                saved_count++;
            }
        }
        
        out.close();
        cout << "Results saved. Total vertices with edges: " << saved_count << endl;
    }
};

// 加载图数据
Graph load_temporal_graph(const string& filepath) {
    Graph G;
    Timer timer;
    
    cout << "Loading graph from " << filepath << "..." << endl;
    
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error: Cannot open input file!" << endl;
        return G;
    }
    
    string line;
    size_t edge_count = 0;
    
    while (getline(file, line)) {
        if (line.empty() || line[0] == '%' || line[0] == '#') continue;
        
        istringstream iss(line);
        vertex_t u, v;
        
        if (iss >> u >> v) {
            G.add_edge(u, v);
            edge_count++;
            
            if (edge_count % 100000 == 0) {
                cout << "  Loaded " << edge_count << " edges..." << endl;
            }
        }
    }
    
    file.close();
    
    double elapsed = timer.elapsed_seconds();
    cout << "Graph loaded in " << fixed << setprecision(3) << elapsed << " seconds" << endl;
    cout << "  Nodes: " << G.num_vertices << endl;
    cout << "  Edges: " << G.num_edges << endl;
    cout << "  Self-loops skipped: " << G.self_loops_skipped << endl;
    
    return G;
}

int main() {
    cout << "============================================================" << endl;
    cout << "Experiment 1: C++ BZ Algorithm K-core Computation" << endl;
    cout << "============================================================" << endl;
    
    // 设置文件路径
    string dataset_path = "/home/jding/dataset/sx-superuser.txt";
    string output_file = "/home/jding/e1_bz_cpp_results.txt";
    
    // 加载图
    Graph G = load_temporal_graph(dataset_path);
    
    if (G.num_vertices == 0) {
        cerr << "Error: Empty graph!" << endl;
        return 1;
    }
    
    // 创建BZ算法实例并计算
    BZAlgorithm bz(G);
    
    Timer total_timer;
    bz.compute();
    double total_time = total_timer.elapsed_seconds();
    
    // 分析结果
    bz.analyze_distribution();
    
    // 保存结果
    bz.save_results(output_file);
    
    // 性能总结
    cout << "\n============================================================" << endl;
    cout << "Performance Summary:" << endl;
    cout << "  Total computation time: " << fixed << setprecision(3) 
         << total_time << " seconds" << endl;
    cout << "  Vertices processed: " << G.num_vertices << endl;
    cout << "  Edges processed: " << G.num_edges << endl;
    cout << "  Throughput: " << fixed << setprecision(0) 
         << G.num_edges / total_time << " edges/second" << endl;
    cout << "============================================================" << endl;
    
    return 0;
}