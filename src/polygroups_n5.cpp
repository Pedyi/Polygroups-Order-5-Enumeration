// ===================================================================================
//  FINAL HYBRID CODE for n=5  (revised version, accompanying the DM-GAA revision)
//
//  Mathematical remarks used throughout:
//   * In a polygroup the inverse map is an INVOLUTION: (x^-1)^-1 = x.  Hence on the
//     four non-identity elements it has cycle type 1+1+1+1, 2+1+1 or 2+2 only;
//     3-cycles and 4-cycles are impossible and are never searched.
//   * Isomorphisms fix the identity (it is the unique scalar identity), so two tables with
//     identity = element 1 are isomorphic iff they differ by a permutation of {2,3,4,5}.
// ===================================================================================

#include <iostream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <omp.h>
#include <set>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <thread>
#include <atomic>
#include <fstream>
#include <mutex>
#include <numeric>
#include <map>
#include <string>

// --- Configuration ---
constexpr int N = 5;
// ---------------------

// --- consts ---
constexpr int BITS_PER_CELL = 5; 
constexpr int NUM_NON_IDENTITY = N - 1;
constexpr int NUM_CELLS_TO_FILL = NUM_NON_IDENTITY * NUM_NON_IDENTITY;
constexpr int CELLS_IN_SUBGROUP_N4 = (N-2)*(N-2);
const std::string CHECKPOINT_FILE = "polygroup_checkpoint_n5.bin";
const std::string FINAL_RESULTS_FILE = "polygroups_final_results.csv";


// --- define variables ---
using Subset = uint16_t;
using HyperTable = std::vector<std::vector<Subset>>;
using EncodedTable = __uint128_t;

struct BasePolygoup {
    std::string name;
    HyperTable table;
    std::vector<int> inverse_map_part;
};

// --- global variables ---
std::mutex g_io_mutex;
std::atomic<uint64_t> g_structures_checked = 0;
std::atomic<size_t> g_candidates_found_count = 0;
std::set<EncodedTable> g_candidate_cache;
std::chrono::time_point<std::chrono::high_resolution_clock> g_total_start_time;

// --- call functions ---
void save_checkpoint(int last_case_done, const std::set<EncodedTable>& keys);
int load_checkpoint(std::set<EncodedTable>& keys);
std::vector<std::vector<int>> get_permutations(int num_elements);
EncodedTable encode_to_128bit(const HyperTable& table);
void decode_from_128bit(EncodedTable key, HyperTable& table_to_fill);
std::string bitmask_to_string(Subset s);
std::pair<int, int> choose_next_cell(const HyperTable& table, const std::vector<std::pair<int, int>>& remaining_cells);
EncodedTable apply_permutation_and_encode(const HyperTable& table, const std::vector<int>& p_in);
EncodedTable apply_permutation_and_encode_generic(const HyperTable& table, int n_size, const std::vector<int>& p_in);
bool check_P3(const HyperTable& table, const std::vector<int>& inverse_map_indices);
bool check_associativity(const HyperTable& table);
bool check_full_axioms(const HyperTable& table, const std::vector<int>& inverse_map_part_labels);
bool is_partially_associative(const HyperTable& table);
void populate_p2_database(std::vector<BasePolygoup>& db);
void populate_p3_database(std::vector<BasePolygoup>& db);
void populate_p4_database(std::vector<BasePolygoup>& db);
void run_constructive_phase(std::set<EncodedTable>& unique_keys);
std::set<EncodedTable> get_order_4_canonical_keys();
std::string get_cycle_structure(const std::vector<int>& p_part);
void recursive_completion_search_intelligent(HyperTable table, std::vector<std::pair<int, int>> remaining_cells, const std::vector<int>& inverse_map_part_labels, std::vector<HyperTable>& solutions, const std::set<EncodedTable>& o4_keys, int current_case_idx);
void run_completion_phase(std::set<EncodedTable>& unique_keys, const std::set<EncodedTable>& o4_keys, int start_idx);
void save_progress_safely(int current_case_idx, const std::set<EncodedTable>& keys, bool force_save);
void append_result_to_csv(const EncodedTable& key, int structure_count);
uint16_t compute_required_elements(const HyperTable& table, int r, int c, const std::vector<int>& full_inverse_map_indices);
bool check_P3_for_cell(const HyperTable& table, const std::vector<int>& inverse_map, int y, int z);


std::ostream& operator<<(std::ostream& os, __uint128_t val) {
    if (val == 0) return os << "0";
    std::string s = "";
    while (val > 0) { s += std::to_string((int)(val % 10)); val /= 10; }
    std::reverse(s.begin(), s.end());
    return os << s;
}

// ===================================================================================
//  Storage, auxiliary, and principle checking functions
// ===================================================================================

std::string bitmask_to_string(Subset s) {
    if (s == 0) return "{}";
    std::string result = "{";
    bool first = true;
    for (int i = 0; i < N; ++i) {
        if ((s >> i) & 1) {
            if (!first) result += ",";
            result += std::to_string(i + 1);
            first = false;
        }
    }
    result += "}";
    return result;
}

void append_result_to_csv(const EncodedTable& key, int structure_count) {
    std::ofstream csv_file(FINAL_RESULTS_FILE, std::ios::app);
    if (!csv_file) {
        std::cerr << "Error: Could not open file for appending: " << FINAL_RESULTS_FILE << std::endl;
        return;
    }
    csv_file << "Unique Structure #" << structure_count << ",Key: " << key << "\n";
    HyperTable table(N, std::vector<Subset>(N, 0));
    for (int r=0; r<N; ++r) { table[r][0] = table[0][r] = (1<<r); }
    decode_from_128bit(key, table);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            csv_file << "\"" << bitmask_to_string(table[i][j]) << "\"" << (j < N - 1 ? "," : "");
        }
        csv_file << "\n";
    }
    csv_file << "\n";
}

void save_checkpoint(int last_case_done, const std::set<EncodedTable>& keys) {
    std::ofstream out(CHECKPOINT_FILE, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "Error: Could not open checkpoint file for writing: " << CHECKPOINT_FILE << std::endl;
        return;
    }
    out.write(reinterpret_cast<const char*>(&last_case_done), sizeof(last_case_done));
    uint64_t checks = g_structures_checked.load();
    out.write(reinterpret_cast<const char*>(&checks), sizeof(checks));
    size_t candidates_found = g_candidates_found_count.load();
    out.write(reinterpret_cast<const char*>(&candidates_found), sizeof(candidates_found));
    size_t num_keys = keys.size();
    out.write(reinterpret_cast<const char*>(&num_keys), sizeof(num_keys));
    for (const auto& key : keys) {
        out.write(reinterpret_cast<const char*>(&key), sizeof(EncodedTable));
    }
    size_t num_cache_keys = g_candidate_cache.size();
    out.write(reinterpret_cast<const char*>(&num_cache_keys), sizeof(num_cache_keys));
    for (const auto& key : g_candidate_cache) {
        out.write(reinterpret_cast<const char*>(&key), sizeof(EncodedTable));
    }
}

void save_progress_safely(int current_case_idx, const std::set<EncodedTable>& keys, bool force_save = false) {
    const uint64_t save_interval = 20'000'000;
    static std::atomic<uint64_t> last_saved_check_count = 0;
    uint64_t current_checks = g_structures_checked.load(std::memory_order_relaxed);
    if (force_save || (current_checks - last_saved_check_count > save_interval)) {
        if (g_io_mutex.try_lock()) {
            current_checks = g_structures_checked.load(std::memory_order_relaxed);
            if (force_save || (current_checks - last_saved_check_count > save_interval)) {
                int last_completed_idx = current_case_idx;
                if (!force_save && current_case_idx >= 0) { 
                    last_completed_idx = current_case_idx - 1;
                }
                save_checkpoint(last_completed_idx, keys);
                last_saved_check_count = current_checks;
                size_t candidates_count = g_candidates_found_count.load(std::memory_order_relaxed);
                size_t unique_count = keys.size();
                std::cout << "States checked: " << current_checks 
                          << " | Candidates found: " << candidates_count 
                          << " | Unique structures: " << unique_count << "\r" << std::flush;
            }
            g_io_mutex.unlock();
        }
    }
}

int load_checkpoint(std::set<EncodedTable>& keys) {
    std::ifstream in(CHECKPOINT_FILE, std::ios::binary);
    if (!in) {
        std::cout << "[INFO] No checkpoint file found. Starting from scratch." << std::endl;
        return -1;
    }
    int last_case_done;
    in.read(reinterpret_cast<char*>(&last_case_done), sizeof(last_case_done));
    if (in.eof()) {
         std::cout << "[INFO] Checkpoint file is empty or corrupted. Starting from scratch." << std::endl;
         return -1;
    }
    uint64_t checks;
    in.read(reinterpret_cast<char*>(&checks), sizeof(checks));
    g_structures_checked = checks;
    size_t candidates_found;
    in.read(reinterpret_cast<char*>(&candidates_found), sizeof(candidates_found));
    g_candidates_found_count = candidates_found;
    size_t num_keys;
    in.read(reinterpret_cast<char*>(&num_keys), sizeof(num_keys));
    keys.clear();
    for (size_t i = 0; i < num_keys; ++i) {
        EncodedTable key;
        in.read(reinterpret_cast<char*>(&key), sizeof(EncodedTable));
        keys.insert(key);
    }
    std::cout << "[INFO] Checkpoint file loaded." << std::endl;
    std::cout << "  - Resuming after inverse case #" << last_case_done + 1 << std::endl;
    std::cout << "  - Loaded " << keys.size() << " unique structures." << std::endl;
    std::cout << "  - Resuming with " << g_structures_checked.load() << " states checked." << std::endl;
    size_t num_cache_keys;
    in.read(reinterpret_cast<char*>(&num_cache_keys), sizeof(num_cache_keys));
    if(in.good()) {
        g_candidate_cache.clear();
        for (size_t i = 0; i < num_cache_keys; ++i) {
            EncodedTable key;
            in.read(reinterpret_cast<char*>(&key), sizeof(EncodedTable));
            g_candidate_cache.insert(key);
        }
        std::cout << "  - Loaded " << g_candidate_cache.size() << " cached candidates." << std::endl;
    }
    return last_case_done;
}

std::pair<int, int> choose_next_cell(const HyperTable& table, const std::vector<std::pair<int, int>>& remaining_cells) {
    if (remaining_cells.empty()) { return {-1, -1}; }
    std::pair<int, int> best_cell = remaining_cells[0];
    int max_score = -1;
    std::vector<int> row_counts(N, 0);
    std::vector<int> col_counts(N, 0);
    for(int i = 1; i < N; ++i) {
        for(int j = 1; j < N; ++j) {
            if (table[i][j] != 0) {
                row_counts[i]++;
                col_counts[j]++;
            }
        }
    }
    for (const auto& cell : remaining_cells) {
        int r = cell.first;
        int c = cell.second;
        int current_score = row_counts[r] + col_counts[c];
        if (current_score > max_score) {
            max_score = current_score;
            best_cell = cell;
        }
    }
    return best_cell;
}

std::vector<std::vector<int>> get_permutations(int num_elements) {
    std::vector<int> p(num_elements);
    std::iota(p.begin(), p.end(), 1);
    std::vector<std::vector<int>> permutations_list;
    do {
        permutations_list.push_back(p);
    } while (std::next_permutation(p.begin(), p.end()));
    return permutations_list;
}

void decode_from_128bit(EncodedTable key, HyperTable& table_to_fill) {
    int shift = 0;
    for (int i = 1; i < N; ++i) for (int j = 1; j < N; ++j) {
        table_to_fill[i][j] = (key >> shift) & ((1 << BITS_PER_CELL) - 1);
        shift += BITS_PER_CELL;
    }
}

EncodedTable encode_to_128bit(const HyperTable& table) {
    EncodedTable key = 0;
    int shift = 0;
    for (int i = 1; i < N; ++i) {
        for (int j = 1; j < N; ++j) {
            key |= (EncodedTable)table[i][j] << shift;
            shift += BITS_PER_CELL;
        }
    }
    return key;
}

EncodedTable apply_permutation_and_encode_generic(const HyperTable& table, int n_size, const std::vector<int>& p_in) {
    HyperTable p_table(n_size, std::vector<Subset>(n_size, 0));
    std::vector<int> p(n_size);
    p[0] = 0;
    for (size_t i = 0; i < p_in.size(); ++i) p[i + 1] = p_in[i];
    std::vector<int> p_inv(n_size);
    for(int i=0; i<n_size; ++i) p_inv[p[i]] = i;
    for (int i = 0; i < n_size; ++i) {
        for (int j = 0; j < n_size; ++j) {
            Subset original_set = table[p[i]][p[j]];
            Subset permuted_set = 0;
            for (int element_idx = 0; element_idx < n_size; ++element_idx) {
                if ((original_set >> element_idx) & 1) {
                    permuted_set |= (1 << p_inv[element_idx]);
                }
            }
            p_table[i][j] = permuted_set;
        }
    }
    EncodedTable key = 0;
    int shift = 0;
    for (int i = 1; i < n_size; ++i) {
        for (int j = 1; j < n_size; ++j) {
            key |= (EncodedTable)p_table[i][j] << shift;
            shift += BITS_PER_CELL;
        }
    }
    return key;
}

EncodedTable apply_permutation_and_encode(const HyperTable& table, const std::vector<int>& p_in) {
    return apply_permutation_and_encode_generic(table, N, p_in);
}

bool check_P3(const HyperTable& table, const std::vector<int>& inverse_map_indices) {
    for (int x=0; x<N; ++x) for (int y=0; y<N; ++y) for (int z=0; z<N; ++z) {
        if ((table[y][z] >> x) & 1) {
            if (!((table[x][inverse_map_indices[z]] >> y) & 1)) return false;
            if (!((table[inverse_map_indices[y]][x] >> z) & 1)) return false;
        }
    }
    return true;
}


bool check_P3_for_cell(const HyperTable& table, const std::vector<int>& inverse_map, int y, int z) {
    Subset members = table.at(y).at(z);
    for (int x = 0; x < N; ++x) {
        if ((members >> x) & 1) {
            int inv_z = inverse_map.at(z);
            if (table.at(x).at(inv_z) != 0 && !((table.at(x).at(inv_z) >> y) & 1)) {
                return false;
            }
            int inv_y = inverse_map.at(y);
            if (table.at(inv_y).at(x) != 0 && !((table.at(inv_y).at(x) >> z) & 1)) {
                return false;
            }
        }
    }
    return true;
}

bool check_associativity(const HyperTable& table) {
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) for (int l = 0; l < N; ++l) {
        Subset lhs = 0, rhs = 0;
        for (int s = 0; s < N; ++s) if ((table[i][j] >> s) & 1) lhs |= table[s][l];
        for (int t = 0; t < N; ++t) if ((table[j][l] >> t) & 1) rhs |= table[i][t];
        if (lhs != rhs) return false;
    }
    return true;
}

bool check_full_axioms(const HyperTable& table, const std::vector<int>& inverse_map_part_labels) {
    std::vector<int> full_inverse_map_indices(N);
    full_inverse_map_indices[0] = 0; // inv(element 1) is element 1 -> index 0 to index 0
    for(size_t i=0; i<inverse_map_part_labels.size(); ++i) {
        full_inverse_map_indices[i + 1] = inverse_map_part_labels[i] - 1;
    }
    if (!check_P3(table, full_inverse_map_indices)) return false;
    if (!check_associativity(table)) return false;
    return true;
}

bool is_partially_associative(const HyperTable& table) {
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) for (int l = 0; l < N; ++l) {
        if (table[i][j] == 0 || table[j][l] == 0) continue;
        Subset lhs_members = table[i][j];
        bool lhs_ready = true;
        for (int s = 0; s < N; ++s) {
            if (((lhs_members >> s) & 1) && table[s][l] == 0) {
                lhs_ready = false;
                break;
            }
        }
        if (!lhs_ready) continue;
        Subset rhs_members = table[j][l];
        bool rhs_ready = true;
        for (int t = 0; t < N; ++t) {
            if (((rhs_members >> t) & 1) && table[i][t] == 0) {
                rhs_ready = false;
                break;
            }
        }
        if (!rhs_ready) continue;
        Subset lhs_result = 0, rhs_result = 0;
        for (int s = 0; s < N; ++s) if ((lhs_members >> s) & 1) lhs_result |= table[s][l];
        for (int t = 0; t < N; ++t) if ((rhs_members >> t) & 1) rhs_result |= table[i][t];
        if (lhs_result != rhs_result) return false;
    }
    return true;
}

uint16_t compute_required_elements(const HyperTable& table, int r, int c, const std::vector<int>& full_inverse_map_indices) {
    uint16_t required_mask = 0;
    for (int y = 0; y < N; ++y) {
        for (int z = 0; z < N; ++z) {
            if (table[y][z] == 0) continue; 
            Subset s_yz = table[y][z];
            int inv_z = full_inverse_map_indices[z];
            if (c == inv_z) { 
                if ((s_yz >> r) & 1) {
                    required_mask |= (1 << y);
                }
            }
            int inv_y = full_inverse_map_indices[y];
            if (r == inv_y) { 
                if ((s_yz >> c) & 1) {
                    required_mask |= (1 << z);
                }
            }
        }
    }
    return required_mask;
}

// ===================================================================================
// Phase 1: constructive Search
// ===================================================================================
void construct_recursive_intelligent(HyperTable table, std::vector<std::pair<int, int>> remaining_cells, 
                                     const std::vector<int>& inverse_map, std::vector<HyperTable>& solutions, int current_case_idx);
void search_from_base(const BasePolygoup& base, const std::vector<int>& new_inv_map_part, std::vector<HyperTable>& solutions) {
    HyperTable initial_table(N, std::vector<Subset>(N, 0));
    std::vector<int> full_inv_map(N);
    
    full_inv_map[0] = 0;
    int base_n = base.table.size();
    for(int i=0; i<base_n-1; ++i) full_inv_map[i+1] = base.inverse_map_part[i];
    for(size_t i=0; i<new_inv_map_part.size(); ++i) full_inv_map[base_n+i] = new_inv_map_part[i];

    for(int r=0; r<base_n; ++r) for(int c=0; c<base_n; ++c) initial_table[r][c] = base.table[r][c];
    for(int r=0; r<N; ++r) { initial_table[0][r] = (1 << r); initial_table[r][0] = (1 << r); }

    
    
    std::vector<std::pair<int, int>> cells_to_fill;
    for (int r = 1; r < N; ++r) {
        for (int c = 1; c < N; ++c) {
            if (initial_table[r][c] == 0) {
                cells_to_fill.push_back({r, c});
            }
        }
    }
    
    if (cells_to_fill.empty()) {
        if (check_P3(initial_table, full_inv_map) && check_associativity(initial_table)) {
            solutions.push_back(initial_table);
        }
        return;
    }

    std::pair<int, int> first_cell = choose_next_cell(initial_table, cells_to_fill);
    
    std::vector<std::pair<int, int>> after_first_cells;
    for(const auto& cell : cells_to_fill) {
        if (cell != first_cell) {
            after_first_cells.push_back(cell);
        }
    }

    int r_first = first_cell.first;
    int c_first = first_cell.second;

    #pragma omp parallel
    {
        std::vector<HyperTable> thread_local_solutions;
        
        #pragma omp for schedule(dynamic)
        for (uint32_t extra_bits = 0; extra_bits < (1 << N); ++extra_bits) {
            if ((extra_bits & initial_table[r_first][c_first]) != 0) continue;
            
            Subset s = initial_table[r_first][c_first] | extra_bits;
            if (s == 0) continue;

            bool contains_e = (s & 1);
            bool is_inverse_cell = (c_first == full_inv_map[r_first]);
            if (contains_e != is_inverse_cell) continue;

            HyperTable start_table = initial_table;
            start_table[r_first][c_first] = s;
            
            g_structures_checked++;
            save_progress_safely(-1, {}, false);

            if (check_P3_for_cell(start_table, full_inv_map, r_first, c_first) && is_partially_associative(start_table)) {
                construct_recursive_intelligent(start_table, after_first_cells, full_inv_map, thread_local_solutions, -1);
            }
        }
        
        #pragma omp critical
        {
            solutions.insert(solutions.end(), thread_local_solutions.begin(), thread_local_solutions.end());
        }
    }
}

void construct_recursive_intelligent(HyperTable table, std::vector<std::pair<int, int>> remaining_cells, 
                                     const std::vector<int>& inverse_map, std::vector<HyperTable>& solutions, int current_case_idx = -1) {
    if (remaining_cells.empty()) {
        if (check_P3(table, inverse_map) && check_associativity(table)) {
            solutions.push_back(table);
        }
        return;
    }

    std::pair<int, int> current_cell = choose_next_cell(table, remaining_cells);
    int i = current_cell.first;
    int j = current_cell.second;

    std::vector<std::pair<int, int>> next_remaining_cells;
    for(const auto& cell : remaining_cells) {
        if (cell.first != i || cell.second != j) {
            next_remaining_cells.push_back(cell);
        }
    }
    
    HyperTable current_table_state = table; 
    uint16_t required_mask = compute_required_elements(current_table_state, i, j, inverse_map);

    for (uint32_t extra_bits = 0; extra_bits < (1 << N); ++extra_bits) {
        if ((extra_bits & current_table_state.at(i).at(j)) != 0) continue;
        
        Subset s = current_table_state.at(i).at(j) | extra_bits;

        if ((s & required_mask) != required_mask) continue;

        if (s == 0) continue;

        bool contains_e = (s & 1);
        bool is_inverse_cell = (j == inverse_map[i]);
        if (contains_e != is_inverse_cell) continue;

        HyperTable next_table = current_table_state;
        next_table[i][j] = s;
        
        g_structures_checked++;
        save_progress_safely(current_case_idx, {}, false);

        if (!check_P3_for_cell(next_table, inverse_map, i, j)) continue;
        if (!is_partially_associative(next_table)) continue;
        
        construct_recursive_intelligent(next_table, next_remaining_cells, inverse_map, solutions, current_case_idx);
    }
}

void run_constructive_phase(std::set<EncodedTable>& unique_keys) {
    auto phase_start_time = std::chrono::high_resolution_clock::now();
    std::cout << "=========================================================\n"
              << "  PHASE 1: CONSTRUCTIVE SEARCH (Extending N < 5)\n"
              << "=========================================================" << std::endl;
    
    std::vector<BasePolygoup> p2_db, p3_db, p4_db;
    populate_p2_database(p2_db);
    populate_p3_database(p3_db);
    populate_p4_database(p4_db);

    std::vector<HyperTable> all_candidates;
    auto perms_n_minus_1 = get_permutations(NUM_NON_IDENTITY);

    std::cout << "\n--- Extending from N=2 Polygroups ---" << std::endl;
    // New elements are 2,3,4 (0-based).  Inverse maps on them must be involutions; up to relabelling
    // of the new elements there are exactly two: identity {2,3,4} and one transposition {2,4,3}.
    std::vector<std::vector<int>> p2_ext_invs = {{2,3,4}, {2,4,3}};
    for(const auto& base : p2_db) {
        for(const auto& inv : p2_ext_invs) {
            std::vector<HyperTable> solutions;
            search_from_base(base, inv, solutions);
            all_candidates.insert(all_candidates.end(), solutions.begin(), solutions.end());
        }
    }
    
    std::cout << "\n--- Extending from N=3 Polygroups ---" << std::endl;
    // New elements are 3,4: identity {3,4} or the transposition {4,3}.
    std::vector<std::vector<int>> p3_ext_invs = {{4,3}, {3,4}};
    for(const auto& base : p3_db) {
        for(const auto& inv : p3_ext_invs) {
            std::vector<HyperTable> solutions;
            search_from_base(base, inv, solutions);
            all_candidates.insert(all_candidates.end(), solutions.begin(), solutions.end());
        }
    }

    std::cout << "\n--- Extending from N=4 Polygroups ---" << std::endl;
    for(const auto& base : p4_db) {
        std::vector<HyperTable> solutions;
        search_from_base(base, {4}, solutions);
        all_candidates.insert(all_candidates.end(), solutions.begin(), solutions.end());
    }

    std::cout << "\n--- Filtering Phase 1 Results (" << all_candidates.size() << " candidates) ---" << std::endl;
    
    #pragma omp parallel
    {
        std::set<EncodedTable> thread_local_keys;
        #pragma omp for schedule(dynamic)
        for(size_t i = 0; i < all_candidates.size(); ++i) {
            const auto& table = all_candidates[i];
            
            EncodedTable smallest_key = encode_to_128bit(table);
            for (const auto& p : perms_n_minus_1) {
                smallest_key = std::min(smallest_key, apply_permutation_and_encode(table, p));
            }
            thread_local_keys.insert(smallest_key);
        }

        #pragma omp critical
        {
            for (const auto& key : thread_local_keys) {
                auto insert_result = unique_keys.insert(key);
                if (insert_result.second) { 
                    append_result_to_csv(key, unique_keys.size());
                    save_progress_safely(-1, unique_keys, true);
                }
            }
        }
    }

    auto phase_end_time = std::chrono::high_resolution_clock::now();
    std::cout << "\n--- PHASE 1 SUMMARY ---" << std::endl;
    std::cout << "Found " << unique_keys.size() << " unique structures." << std::endl;
    std::cout << "Phase 1 duration: " << std::chrono::duration<double>(phase_end_time - phase_start_time).count() << "s." << std::endl;
}


// ===================================================================================
// phase 2 : COMPLETION SEARCH
// ===================================================================================

void recursive_completion_search_intelligent(HyperTable table, std::vector<std::pair<int, int>> remaining_cells,
                                         const std::vector<int>& inverse_map_part_labels, std::vector<HyperTable>& solutions, 
                                         const std::set<EncodedTable>& o4_keys, int current_case_idx) {
    
    if (remaining_cells.empty()) {
        if (check_full_axioms(table, inverse_map_part_labels)) {
            solutions.push_back(table);
        }
        return;
    }

    if (remaining_cells.size() <= (NUM_CELLS_TO_FILL - CELLS_IN_SUBGROUP_N4)) {
        bool subgrid_complete = true;
        for(int r=1; r<N-1; ++r) for(int c=1; c<N-1; ++c){
            if(table[r][c] == 0) {
                subgrid_complete = false; break;
            }
        }

        if(subgrid_complete) {
            for(int i=1; i<N-1; ++i) for(int j=1; j<N-1; ++j) {
                if ((table[i][j] >> (N-1)) & 1) goto continue_search; 
            }
            HyperTable sub_table(N - 1, std::vector<Subset>(N - 1));
            for(int i=0; i<N-1; ++i) for(int j=0; j<N-1; ++j) sub_table[i][j] = table[i][j];
            auto sub_perms = get_permutations(N-2);
            EncodedTable smallest_key_n4 = apply_permutation_and_encode_generic(sub_table, N - 1, sub_perms[0]);
            for(const auto& p : sub_perms){
                smallest_key_n4 = std::min(smallest_key_n4, apply_permutation_and_encode_generic(sub_table, N-1, p));
            }
            if (o4_keys.count(smallest_key_n4)) return;
        }
    }
    continue_search:;

    std::pair<int, int> current_cell = choose_next_cell(table, remaining_cells);
    int i = current_cell.first;
    int j = current_cell.second;

    std::vector<std::pair<int, int>> next_remaining_cells;
    for(const auto& cell : remaining_cells) {
        if (cell.first != i || cell.second != j) {
            next_remaining_cells.push_back(cell);
        }
    }
    
    std::vector<int> full_inv_map_indices(N);
    full_inv_map_indices[0] = 0;
    for(size_t idx=0; idx < inverse_map_part_labels.size(); ++idx) {
        full_inv_map_indices[idx + 1] = inverse_map_part_labels[idx] - 1;
    }
    
    HyperTable current_table_state = table;
    
    uint16_t required_mask = compute_required_elements(current_table_state, i, j, full_inv_map_indices);

    for (Subset s = 1; s < (1 << N); ++s) {
        if ((s & required_mask) != required_mask) continue;

        g_structures_checked++;
        if(current_case_idx != -1) save_progress_safely(current_case_idx, {}, false);
        
        bool contains_e = (s & 1);
        bool is_inverse_cell = (j == full_inv_map_indices[i]);
        if (contains_e != is_inverse_cell) continue;
        
        HyperTable next_table = current_table_state;
        next_table[i][j] = s;

        if (!is_partially_associative(next_table)) continue;
        
        recursive_completion_search_intelligent(next_table, next_remaining_cells, inverse_map_part_labels, solutions, o4_keys, current_case_idx);
    }
}

void search_parallel_entry_completion(const std::vector<int>& inverse_map_part_labels, std::vector<HyperTable>& solutions, 
                                      const std::set<EncodedTable>& o4_keys, int current_case_idx) {
    
    HyperTable initial_table(N, std::vector<Subset>(N, 0));
    for (int r = 0; r < N; ++r) { 
        initial_table[0][r] = (1 << r); 
        initial_table[r][0] = (1 << r); 
    }

    std::vector<std::pair<int,int>> initial_remaining_cells;
    for(int r=1; r<N; ++r) for(int c=1; c<N; ++c) {
        initial_remaining_cells.push_back({r,c});
    }

    std::pair<int, int> first_cell = choose_next_cell(initial_table, initial_remaining_cells);
    
    std::vector<std::pair<int, int>> after_first_cells;
    for (const auto& cell : initial_remaining_cells) {
        if (cell != first_cell) {
            after_first_cells.push_back(cell);
        }
    }

    int r_first = first_cell.first;
    int c_first = first_cell.second;

    std::vector<int> full_inv_map_indices(N);
    full_inv_map_indices[0] = 0;
    for(size_t idx=0; idx < inverse_map_part_labels.size(); ++idx) {
        full_inv_map_indices[idx + 1] = inverse_map_part_labels[idx] - 1;
    }

    #pragma omp parallel
    {
        std::vector<HyperTable> local_solutions;
        
        #pragma omp for schedule(dynamic)
        for (Subset s = 1; s < (1 << N); ++s) {
            bool is_inverse_cell = (c_first == full_inv_map_indices[r_first]);
            if (is_inverse_cell != (bool)(s & 1)) continue;

            HyperTable start_table = initial_table;
            start_table[r_first][c_first] = s;
            
            if (is_partially_associative(start_table)) {
                recursive_completion_search_intelligent(start_table, after_first_cells, inverse_map_part_labels, local_solutions, o4_keys, current_case_idx);
            }
        }

        #pragma omp critical
        {
            solutions.insert(solutions.end(), local_solutions.begin(), local_solutions.end());
        }
    }
}


std::string get_cycle_structure(const std::vector<int>& p_part) {
    std::map<int, int> p_map;
    for(size_t i = 0; i < p_part.size(); ++i) {
        p_map[i + 1] = p_part[i];
    }
    std::vector<bool> visited(p_part.size() + 1, false);
    std::vector<int> lengths;
    for (unsigned int i = 1; i <= p_part.size(); ++i) {
        if (!visited[i]) {
            int current_len = 0;
            int j = i;
            while (!visited[j]) {
                visited[j] = true;
                j = p_map[j];
                current_len++;
            }
            lengths.push_back(current_len);
        }
    }
    std::sort(lengths.rbegin(), lengths.rend());
    std::stringstream ss;
    for(size_t i = 0; i < lengths.size(); ++i) { 
        ss << lengths[i] << (i == lengths.size() - 1 ? "" : "_"); 
    }
    return ss.str();
}

void run_completion_phase(std::set<EncodedTable>& unique_keys, const std::set<EncodedTable>& o4_keys, int start_idx) {
    auto phase_start_time = std::chrono::high_resolution_clock::now();
    std::cout << "\n=========================================================\n"
              << "  PHASE 2: COMPLETION SEARCH (Finding Primitives)\n"
              << "=========================================================" << std::endl;

    auto all_perms_iso = get_permutations(NUM_NON_IDENTITY);
    std::map<std::string, std::vector<int>> representative_inverses_map;
    
    for(const auto& p : all_perms_iso) {
        // The inverse map of a polygroup is an involution, so only permutations with p(p(i)) = i are admissible.
        bool is_involution = true;
        for (size_t i = 0; i < p.size(); ++i) if (p[p[i] - 1] != (int)i + 1) { is_involution = false; break; }
        if (!is_involution) continue;
        std::string structure = get_cycle_structure(p);
        if (representative_inverses_map.find(structure) == representative_inverses_map.end()) {
            std::vector<int> inv_map_labels(NUM_NON_IDENTITY);
            for(size_t i = 0; i < p.size(); ++i) {
                for(size_t j = 0; j < p.size(); ++j) {
                    if (p[j] == i + 1) {
                        inv_map_labels[i] = j + 2;
                        break;
                    }
                }
            }
            representative_inverses_map[structure] = inv_map_labels;
        }
    }

    std::vector<std::pair<std::string, std::vector<int>>> representative_inverses;
    for(const auto& pair : representative_inverses_map) {
        representative_inverses.push_back(pair);
    }
    std::sort(representative_inverses.begin(), representative_inverses.end());

    std::cout << "Generated " << representative_inverses.size() << " representative inverse cases for N=5." << std::endl;
    
    for(int i = start_idx + 1; i < representative_inverses.size(); ++i) {
        const auto& case_type = representative_inverses[i].first;
        const auto& inv_map = representative_inverses[i].second;

        std::cout << "\n--- Starting search for inverse case #" << i+1 << " of " << representative_inverses.size() << ": " << case_type << " ---" << std::endl;
        std::vector<HyperTable> potential_solutions;
        
        search_parallel_entry_completion(inv_map, potential_solutions, o4_keys, i);
        
        g_candidates_found_count += potential_solutions.size();

        std::cout << "\nCase " << case_type << " finished. Found " << potential_solutions.size() << " candidates. Filtering..." << std::endl;

        #pragma omp parallel
        {
            std::set<EncodedTable> thread_local_keys;
            #pragma omp for schedule(dynamic)
            for(size_t j = 0; j < potential_solutions.size(); ++j) {
                const auto& table = potential_solutions[j];
                EncodedTable raw_key = encode_to_128bit(table);
                (void)raw_key;
                {   // every candidate is canonicalised (cheap: 24 relabellings), no skipping

                    EncodedTable smallest_key = raw_key;
                    for (const auto& p_iso : all_perms_iso) {
                        smallest_key = std::min(smallest_key, apply_permutation_and_encode(table, p_iso));
                    }
                    thread_local_keys.insert(smallest_key);
                }
            }

            #pragma omp critical(results_update)
            {
                for (const auto& key : thread_local_keys) {
                    auto insert_result = unique_keys.insert(key);
                    if (insert_result.second) { 
                        append_result_to_csv(key, unique_keys.size());
                        save_progress_safely(i, unique_keys, true);
                    }
                }
            }
        }
        
        save_progress_safely(i, unique_keys, true); 
        auto now = std::chrono::high_resolution_clock::now();
        std::cout << "\n[SUMMARY @ " << (int)(std::chrono::duration<double>(now - g_total_start_time).count()/60.0) 
                  << "min] States checked: " << g_structures_checked.load()
                  << " | Candidates found: " << g_candidates_found_count.load()
                  << " | Unique structures: " << unique_keys.size() << std::endl;
    }

    auto phase_end_time = std::chrono::high_resolution_clock::now();
    std::cout << "\n--- PHASE 2 SUMMARY ---" << std::endl;
    std::cout << "Phase 2 duration (this run): " << std::chrono::duration<double>(phase_end_time - phase_start_time).count() << "s." << std::endl;
}

std::set<EncodedTable> get_order_4_canonical_keys() {
    std::set<EncodedTable> canonical_keys;
    std::vector<BasePolygoup> p4_db;
    populate_p4_database(p4_db);
    auto perms_n_minus_2 = get_permutations(N-2);

    for(const auto& base : p4_db) {
         EncodedTable smallest_key = apply_permutation_and_encode_generic(base.table, N - 1, perms_n_minus_2[0]);
         for (const auto& p : perms_n_minus_2) {
            smallest_key = std::min(smallest_key, apply_permutation_and_encode_generic(base.table, N-1, p));
         }
         canonical_keys.insert(smallest_key);
    }
    return canonical_keys;
}


// ===================================================================================
// main functions
// ===================================================================================
int main() {
    g_total_start_time = std::chrono::high_resolution_clock::now();
    std::cout << "HYBRID POLYGROUP SEARCH ENGINE for N=" << N << std::endl;
    std::cout << "Using " << omp_get_max_threads() << " threads." << std::endl;
    
    std::set<EncodedTable> total_unique_keys;

    int start_case_idx = load_checkpoint(total_unique_keys);

    if (start_case_idx == -1) {
        std::remove(FINAL_RESULTS_FILE.c_str()); 
        run_constructive_phase(total_unique_keys);
    } else {
        std::cout << "[INFO] Skipping Phase 1 due to existing checkpoint." << std::endl;
    }

    std::cout << "\nPreparing for Phase 2: Generating N=4 subgroup keys for pruning..." << std::endl;
    std::set<EncodedTable> order_4_keys = get_order_4_canonical_keys();
    std::cout << "Generated " << order_4_keys.size() << " unique canonical keys for N=4 subgroups." << std::endl;

    run_completion_phase(total_unique_keys, order_4_keys, start_case_idx);

    save_checkpoint(2, total_unique_keys);   // 3 inverse cases: indices 0,1,2

    // ---------------------------------------------------------------------------------
    // Final self-check: every reported table is decoded again and verified from scratch
    // (identity row/column, e in x o y  <=>  y = x^-1, inverse is an involution, P3, full
    // associativity, non-empty cells) and its key must be the minimum over all 24 relabellings.
    // The CSV is then rewritten in increasing key order so that the file is reproducible.
    // ---------------------------------------------------------------------------------
    size_t verified = 0, bad = 0;
    std::map<std::string, size_t> by_inverse_type;
    std::remove(FINAL_RESULTS_FILE.c_str());
    auto all_perms_final = get_permutations(NUM_NON_IDENTITY);
    for (const auto& key : total_unique_keys) {
        HyperTable t(N, std::vector<Subset>(N, 0));
        for (int r = 0; r < N; ++r) { t[r][0] = t[0][r] = (1 << r); }
        decode_from_128bit(key, t);
        std::vector<int> inv(N, -1);
        bool ok = true;
        for (int x = 0; x < N && ok; ++x) {
            int cnt = 0;
            for (int y = 0; y < N; ++y) if (t[x][y] & 1) { inv[x] = y; ++cnt; }
            if (cnt != 1) ok = false;
        }
        for (int x = 0; x < N && ok; ++x) for (int y = 0; y < N; ++y) if (t[x][y] == 0) ok = false;
        for (int x = 0; x < N && ok; ++x) if (inv[inv[x]] != x) ok = false;
        if (ok && !(check_P3(t, inv) && check_associativity(t))) ok = false;
        if (ok) {
            EncodedTable smallest = key;
            for (const auto& p : all_perms_final) smallest = std::min(smallest, apply_permutation_and_encode(t, p));
            if (smallest != key) ok = false;
        }
        if (ok) {
            ++verified;
            int fixed_pts = 0;
            for (int x = 1; x < N; ++x) if (inv[x] == x) ++fixed_pts;
            // an involution on 4 points: 4 fixed points -> 1_1_1_1, 2 -> 2_1_1, 0 -> 2_2
            by_inverse_type[fixed_pts == 4 ? "1_1_1_1" : (fixed_pts == 2 ? "2_1_1" : "2_2")]++;
            append_result_to_csv(key, (int)verified);
        } else { ++bad; }
    }
    std::cout << "\nSelf-check: " << verified << " tables verified, " << bad << " failed." << std::endl;
    for (const auto& kv : by_inverse_type)
        std::cout << "  inverse cycle type " << kv.first << ": " << kv.second << " polygroups" << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "\n\n----------------------------------------" << std::endl;
    std::cout << "           FINAL RESULTS (n=" << N << ")" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total non-isomorphic polygroups found: " << total_unique_keys.size() << std::endl;
    std::cout << "Total structures checked (approx): " << g_structures_checked.load() << std::endl;
    std::cout << "Total execution time: " << std::chrono::duration<double>(end_time - g_total_start_time).count() << "s\n";
    
    return 0;
}

// ===================================================================================
//  Database Definitions
// ===================================================================================
#pragma region "Database Definitions"
void populate_p2_database(std::vector<BasePolygoup>& db) {
    db.push_back({"P_Z2", HyperTable{{1,2},{2,1}}, {1}});
    db.push_back({"P_NonGroup", HyperTable{{1,2},{2,3}}, {1}});
}

void populate_p3_database(std::vector<BasePolygoup>& db) {
    db.push_back({"P4", HyperTable{{1,2,4},{2,1,4},{4,4,3}}, {1, 2}});
    db.push_back({"P5", HyperTable{{1,2,4},{2,1,4},{4,4,7}}, {1, 2}});
    db.push_back({"P6", HyperTable{{1,2,4},{2,3,4},{4,4,3}}, {1, 2}});
    db.push_back({"P7", HyperTable{{1,2,4},{2,3,4},{4,4,7}}, {1, 2}});
    db.push_back({"P8", HyperTable{{1,2,4},{2,5,6},{4,6,3}}, {1, 2}});
    db.push_back({"P9", HyperTable{{1,2,4},{2,5,6},{4,6,7}}, {1, 2}});
    db.push_back({"P10", HyperTable{{1,2,4},{2,7,6},{4,6,7}}, {1, 2}});
    db.push_back({"P11", HyperTable{{1,2,4},{2,2,7},{4,7,4}}, {2, 1}});
    db.push_back({"P12 (Z3)", HyperTable{{1,2,4},{2,4,1},{4,1,2}}, {2, 1}});
    db.push_back({"P13", HyperTable{{1,2,4},{2,6,7},{4,7,6}}, {2, 1}});
}

void populate_p4_database(std::vector<BasePolygoup>& db) {
    // The 102 pairwise non-isomorphic polygroups of order 4 (one table per isomorphism class).
    // Generated by the independent enumerator (verification/enum_indep.cpp) and re-verified in
    // verification/verification.ipynb; the earlier hand-typed list contained typographical errors.
    // Entry layout: {name, 4x4 table (bitmasks, element i <-> bit i-1), inverse map of the 3 non-identity elements (0-based)}.
    db.push_back({"O4_001",{{{1,2,4,8}},{{2,1,4,8}},{{4,4,3,8}},{{8,8,8,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_002",{{{1,2,4,8}},{{2,1,4,8}},{{4,4,3,8}},{{8,8,8,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_003",{{{1,2,4,8}},{{2,1,4,8}},{{4,4,4,15}},{{8,8,15,8}}},std::vector<int>{1,3,2}});
    db.push_back({"O4_004",{{{1,2,4,8}},{{2,1,4,8}},{{4,4,7,8}},{{8,8,8,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_005",{{{1,2,4,8}},{{2,1,4,8}},{{4,4,7,8}},{{8,8,8,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_006",{{{1,2,4,8}},{{2,1,4,8}},{{4,4,8,3}},{{8,8,3,4}}},std::vector<int>{1,3,2}});
    db.push_back({"O4_007",{{{1,2,4,8}},{{2,1,4,8}},{{4,4,11,12}},{{8,8,12,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_008",{{{1,2,4,8}},{{2,1,4,8}},{{4,4,11,12}},{{8,8,12,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_009",{{{1,2,4,8}},{{2,1,4,8}},{{4,4,12,15}},{{8,8,15,12}}},std::vector<int>{1,3,2}});
    db.push_back({"O4_010",{{{1,2,4,8}},{{2,1,4,8}},{{4,4,15,12}},{{8,8,12,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_011",{{{1,2,4,8}},{{2,1,8,4}},{{4,8,1,2}},{{8,4,2,1}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_012",{{{1,2,4,8}},{{2,1,8,4}},{{4,8,2,1}},{{8,4,1,2}}},std::vector<int>{1,3,2}});
    db.push_back({"O4_013",{{{1,2,4,8}},{{2,1,8,4}},{{4,8,5,10}},{{8,4,10,5}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_014",{{{1,2,4,8}},{{2,1,8,4}},{{4,8,13,14}},{{8,4,14,13}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_015",{{{1,2,4,8}},{{2,1,8,4}},{{4,8,14,13}},{{8,4,13,14}}},std::vector<int>{1,3,2}});
    db.push_back({"O4_016",{{{1,2,4,8}},{{2,2,2,15}},{{4,2,5,8}},{{8,15,8,8}}},std::vector<int>{3,2,1}});
    db.push_back({"O4_017",{{{1,2,4,8}},{{2,2,4,11}},{{4,4,11,4}},{{8,11,4,8}}},std::vector<int>{3,2,1}});
    db.push_back({"O4_018",{{{1,2,4,8}},{{2,2,4,11}},{{4,4,15,4}},{{8,11,4,8}}},std::vector<int>{3,2,1}});
    db.push_back({"O4_019",{{{1,2,4,8}},{{2,2,4,15}},{{4,6,15,4}},{{8,11,12,8}}},std::vector<int>{3,2,1}});
    db.push_back({"O4_020",{{{1,2,4,8}},{{2,2,6,15}},{{4,6,11,12}},{{8,15,12,8}}},std::vector<int>{3,2,1}});
    db.push_back({"O4_021",{{{1,2,4,8}},{{2,2,6,15}},{{4,6,15,12}},{{8,15,12,8}}},std::vector<int>{3,2,1}});
    db.push_back({"O4_022",{{{1,2,4,8}},{{2,3,4,8}},{{4,4,3,8}},{{8,8,8,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_023",{{{1,2,4,8}},{{2,3,4,8}},{{4,4,3,8}},{{8,8,8,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_024",{{{1,2,4,8}},{{2,3,4,8}},{{4,4,7,8}},{{8,8,8,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_025",{{{1,2,4,8}},{{2,3,4,8}},{{4,4,7,8}},{{8,8,8,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_026",{{{1,2,4,8}},{{2,3,4,8}},{{4,4,8,3}},{{8,8,3,4}}},std::vector<int>{1,3,2}});
    db.push_back({"O4_027",{{{1,2,4,8}},{{2,3,4,8}},{{4,4,11,12}},{{8,8,12,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_028",{{{1,2,4,8}},{{2,3,4,8}},{{4,4,11,12}},{{8,8,12,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_029",{{{1,2,4,8}},{{2,3,4,8}},{{4,4,12,15}},{{8,8,15,12}}},std::vector<int>{1,3,2}});
    db.push_back({"O4_030",{{{1,2,4,8}},{{2,3,4,8}},{{4,4,15,12}},{{8,8,12,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_031",{{{1,2,4,8}},{{2,3,8,12}},{{4,8,5,10}},{{8,12,10,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_032",{{{1,2,4,8}},{{2,3,8,12}},{{4,8,9,14}},{{8,12,14,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_033",{{{1,2,4,8}},{{2,3,8,12}},{{4,8,13,14}},{{8,12,14,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_034",{{{1,2,4,8}},{{2,3,12,12}},{{4,12,2,3}},{{8,12,3,2}}},std::vector<int>{1,3,2}});
    db.push_back({"O4_035",{{{1,2,4,8}},{{2,3,12,12}},{{4,12,3,2}},{{8,12,2,3}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_036",{{{1,2,4,8}},{{2,3,12,12}},{{4,12,6,15}},{{8,12,15,10}}},std::vector<int>{1,3,2}});
    db.push_back({"O4_037",{{{1,2,4,8}},{{2,3,12,12}},{{4,12,7,10}},{{8,12,10,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_038",{{{1,2,4,8}},{{2,3,12,12}},{{4,12,7,10}},{{8,12,10,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_039",{{{1,2,4,8}},{{2,3,12,12}},{{4,12,11,14}},{{8,12,14,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_040",{{{1,2,4,8}},{{2,3,12,12}},{{4,12,11,14}},{{8,12,14,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_041",{{{1,2,4,8}},{{2,3,12,12}},{{4,12,14,15}},{{8,12,15,14}}},std::vector<int>{1,3,2}});
    db.push_back({"O4_042",{{{1,2,4,8}},{{2,3,12,12}},{{4,12,15,14}},{{8,12,14,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_043",{{{1,2,4,8}},{{2,4,1,8}},{{4,1,2,8}},{{8,8,8,7}}},std::vector<int>{2,1,3}});
    db.push_back({"O4_044",{{{1,2,4,8}},{{2,4,1,8}},{{4,1,2,8}},{{8,8,8,15}}},std::vector<int>{2,1,3}});
    db.push_back({"O4_045",{{{1,2,4,8}},{{2,4,14,5}},{{4,14,15,14}},{{8,5,14,4}}},std::vector<int>{3,2,1}});
    db.push_back({"O4_046",{{{1,2,4,8}},{{2,5,6,8}},{{4,6,3,8}},{{8,8,8,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_047",{{{1,2,4,8}},{{2,5,6,8}},{{4,6,3,8}},{{8,8,8,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_048",{{{1,2,4,8}},{{2,5,6,8}},{{4,6,7,8}},{{8,8,8,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_049",{{{1,2,4,8}},{{2,5,6,8}},{{4,6,7,8}},{{8,8,8,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_050",{{{1,2,4,8}},{{2,5,10,12}},{{4,10,9,6}},{{8,12,6,3}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_051",{{{1,2,4,8}},{{2,5,10,12}},{{4,10,9,6}},{{8,12,6,11}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_052",{{{1,2,4,8}},{{2,5,10,12}},{{4,10,9,14}},{{8,12,14,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_053",{{{1,2,4,8}},{{2,5,10,12}},{{4,10,9,14}},{{8,12,14,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_054",{{{1,2,4,8}},{{2,5,10,12}},{{4,10,13,6}},{{8,12,6,11}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_055",{{{1,2,4,8}},{{2,5,10,12}},{{4,10,13,14}},{{8,12,14,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_056",{{{1,2,4,8}},{{2,5,10,12}},{{4,10,13,14}},{{8,12,14,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_057",{{{1,2,4,8}},{{2,5,14,4}},{{4,14,15,14}},{{8,4,14,5}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_058",{{{1,2,4,8}},{{2,5,14,4}},{{4,14,15,14}},{{8,4,14,13}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_059",{{{1,2,4,8}},{{2,5,14,12}},{{4,14,11,6}},{{8,12,6,11}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_060",{{{1,2,4,8}},{{2,5,14,12}},{{4,14,11,14}},{{8,12,14,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_061",{{{1,2,4,8}},{{2,5,14,12}},{{4,14,11,14}},{{8,12,14,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_062",{{{1,2,4,8}},{{2,5,14,12}},{{4,14,15,6}},{{8,12,6,11}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_063",{{{1,2,4,8}},{{2,5,14,12}},{{4,14,15,14}},{{8,12,14,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_064",{{{1,2,4,8}},{{2,5,14,12}},{{4,14,15,14}},{{8,12,14,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_065",{{{1,2,4,8}},{{2,6,7,8}},{{4,7,6,8}},{{8,8,8,7}}},std::vector<int>{2,1,3}});
    db.push_back({"O4_066",{{{1,2,4,8}},{{2,6,7,8}},{{4,7,6,8}},{{8,8,8,15}}},std::vector<int>{2,1,3}});
    db.push_back({"O4_067",{{{1,2,4,8}},{{2,6,12,11}},{{4,12,11,6}},{{8,11,6,12}}},std::vector<int>{3,2,1}});
    db.push_back({"O4_068",{{{1,2,4,8}},{{2,6,12,11}},{{4,12,15,6}},{{8,11,6,12}}},std::vector<int>{3,2,1}});
    db.push_back({"O4_069",{{{1,2,4,8}},{{2,6,12,15}},{{4,14,11,6}},{{8,11,14,12}}},std::vector<int>{3,2,1}});
    db.push_back({"O4_070",{{{1,2,4,8}},{{2,6,12,15}},{{4,14,15,6}},{{8,11,14,12}}},std::vector<int>{3,2,1}});
    db.push_back({"O4_071",{{{1,2,4,8}},{{2,6,14,15}},{{4,14,11,14}},{{8,15,14,12}}},std::vector<int>{3,2,1}});
    db.push_back({"O4_072",{{{1,2,4,8}},{{2,6,14,15}},{{4,14,15,14}},{{8,15,14,12}}},std::vector<int>{3,2,1}});
    db.push_back({"O4_073",{{{1,2,4,8}},{{2,6,15,10}},{{4,15,6,12}},{{8,10,12,7}}},std::vector<int>{2,1,3}});
    db.push_back({"O4_074",{{{1,2,4,8}},{{2,6,15,10}},{{4,15,6,12}},{{8,10,12,15}}},std::vector<int>{2,1,3}});
    db.push_back({"O4_075",{{{1,2,4,8}},{{2,7,6,8}},{{4,6,7,8}},{{8,8,8,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_076",{{{1,2,4,8}},{{2,7,6,8}},{{4,6,7,8}},{{8,8,8,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_077",{{{1,2,4,8}},{{2,7,10,12}},{{4,10,13,6}},{{8,12,6,11}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_078",{{{1,2,4,8}},{{2,7,10,12}},{{4,10,13,14}},{{8,12,14,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_079",{{{1,2,4,8}},{{2,7,10,12}},{{4,10,13,14}},{{8,12,14,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_080",{{{1,2,4,8}},{{2,7,14,4}},{{4,14,15,14}},{{8,4,14,13}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_081",{{{1,2,4,8}},{{2,7,14,12}},{{4,14,7,10}},{{8,12,10,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_082",{{{1,2,4,8}},{{2,7,14,12}},{{4,14,7,10}},{{8,12,10,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_083",{{{1,2,4,8}},{{2,7,14,12}},{{4,14,11,14}},{{8,12,14,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_084",{{{1,2,4,8}},{{2,7,14,12}},{{4,14,11,14}},{{8,12,14,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_085",{{{1,2,4,8}},{{2,7,14,12}},{{4,14,15,14}},{{8,12,14,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_086",{{{1,2,4,8}},{{2,7,14,12}},{{4,14,15,14}},{{8,12,14,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_087",{{{1,2,4,8}},{{2,12,9,14}},{{4,9,10,14}},{{8,14,14,7}}},std::vector<int>{2,1,3}});
    db.push_back({"O4_088",{{{1,2,4,8}},{{2,12,9,14}},{{4,9,10,14}},{{8,14,14,15}}},std::vector<int>{2,1,3}});
    db.push_back({"O4_089",{{{1,2,4,8}},{{2,13,6,10}},{{4,6,11,12}},{{8,10,12,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_090",{{{1,2,4,8}},{{2,13,6,10}},{{4,6,11,12}},{{8,10,12,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_091",{{{1,2,4,8}},{{2,13,6,10}},{{4,6,15,12}},{{8,10,12,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_092",{{{1,2,4,8}},{{2,13,10,6}},{{4,10,14,13}},{{8,6,13,14}}},std::vector<int>{1,3,2}});
    db.push_back({"O4_093",{{{1,2,4,8}},{{2,13,10,14}},{{4,14,14,13}},{{8,6,15,14}}},std::vector<int>{1,3,2}});
    db.push_back({"O4_094",{{{1,2,4,8}},{{2,13,14,14}},{{4,14,11,14}},{{8,14,14,7}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_095",{{{1,2,4,8}},{{2,13,14,14}},{{4,14,11,14}},{{8,14,14,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_096",{{{1,2,4,8}},{{2,13,14,14}},{{4,14,14,15}},{{8,14,15,14}}},std::vector<int>{1,3,2}});
    db.push_back({"O4_097",{{{1,2,4,8}},{{2,13,14,14}},{{4,14,15,14}},{{8,14,14,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_098",{{{1,2,4,8}},{{2,14,7,12}},{{4,7,14,10}},{{8,12,10,15}}},std::vector<int>{2,1,3}});
    db.push_back({"O4_099",{{{1,2,4,8}},{{2,14,7,14}},{{4,15,14,10}},{{8,12,14,15}}},std::vector<int>{2,1,3}});
    db.push_back({"O4_100",{{{1,2,4,8}},{{2,14,14,15}},{{4,14,15,14}},{{8,15,14,14}}},std::vector<int>{3,2,1}});
    db.push_back({"O4_101",{{{1,2,4,8}},{{2,15,6,10}},{{4,6,15,12}},{{8,10,12,15}}},std::vector<int>{1,2,3}});
    db.push_back({"O4_102",{{{1,2,4,8}},{{2,15,14,14}},{{4,14,15,14}},{{8,14,14,15}}},std::vector<int>{1,2,3}});
}
#pragma endregion