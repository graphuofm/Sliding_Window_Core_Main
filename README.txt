Sliding Window Constraint Coreness Maintenance over Temporal Graphs
============================================================

This repository contains the core code, experiments, and analysis for the paper:

"Sliding Window Constraint Coreness Maintenance over Temporal Graphs"
by Jiacheng Ding, Xiaofei Zhang, University of Memphis

Overview
--------
This project provides a reproducible implementation of the Upper-bound Core Ranking (UCR) algorithm and baseline methods for efficient k-core maintenance on large-scale temporal graphs under sliding window constraints. It includes all core C++/Python code, experiment scripts, and analysis tools used in the paper.

Key features:
- UCR, BZ, MCD/PCD and other baseline algorithms
- Sliding window and batch update experiments
- Performance/statistics/visualization scripts
- Minimal working datasets for quick testing

Directory Structure
-------------------
- src/         All core C++/Python source code and experiment scripts
- data/        Datasets (see below), minidataset/ for small test data
- results/     Key experiment results, tables, and logs
- analysis/    Data analysis and visualization scripts
- docs/        Paper, LaTeX tables, documentation
- scripts/     Auxiliary scripts

How to Reproduce the Experiments
--------------------------------
1. Clone the repository
2. Install dependencies (C++17 compiler, Python 3.7+, py-tgx optional)
3. Compile C++ code (e.g. g++ -O2 -std=c++17 e2.cpp -o e2)
4. Prepare datasets:
   - Small test datasets are in data/minidataset/
   - Large real datasets are NOT included. Download from public sources and place in data/
5. Run experiments (e.g. ./e2 data/minidataset/CollegeMsg.txt)
6. Analyze results in results/; use analysis/ scripts for further analysis/visualization

Minimal Dataset Quick Start
--------------------------
You can quickly test the pipeline using the provided minidataset/:
  cd src
  g++ -O2 -std=c++17 e2.cpp -o e2
  ./e2 ../data/minidataset/CollegeMsg.txt

Notes on Datasets
-----------------
- Large datasets are not included. Download and place in data/
- File format: Each line: src dst timestamp
- Small datasets in data/minidataset/ can be used for code testing and demonstration.

Results and Reproducibility
---------------------------
- All key experiment results, tables, and logs are in the results/ directory.
- The code and scripts are organized to match the structure and methodology described in the paper.
- For full reproducibility, use the same dataset versions and follow the experiment settings in the paper.

Contact
-------
For questions or issues, contact:
- Jiacheng Ding: jding2@memphis.edu
- Xiaofei Zhang: xiaofei.zhang@memphis.edu
