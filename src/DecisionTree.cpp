#include "DecisionTree.hpp"
#include <algorithm>
#include <numeric>
#include <set>
#include <map>
#include <limits>
#include <cmath>
#include <random>

// Fonctions utilitaires

static double calculate_mean(const double* y, const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    return std::accumulate(indices.begin(), indices.end(), 0.0,[y](double acc, int idx) {
                        return acc + y[idx]; 
                        }) / indices.size();
}

static double calculate_variance(const double* y, const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    double mean = calculate_mean(y, indices);
    return std::accumulate(indices.begin(), indices.end(), 0.0,[y, mean](double acc, int idx) {
                            double diff = y[idx] - mean; return acc + diff * diff;
                            }) / indices.size();
}

static double calculate_mode(const double* y, const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    std::map<double, int> counts;
    for (int idx : indices) counts[y[idx]]++;

    auto it = std::max_element(counts.begin(), counts.end(),
                               [](const auto& a, const auto& b) { return a.second < b.second; });
    return it != counts.end() ? it->first : 0.0;
}

static double calculate_gini(const double* y, const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    std::map<double, int> counts;
    for (int idx : indices) counts[y[idx]]++;

    double n = static_cast<double>(indices.size());
    double sum_prob_sq = std::accumulate(counts.begin(), counts.end(), 0.0,
                                         [n](double acc, const auto& pair) {
                                             double prob = pair.second / n;
                                             return acc + prob * prob;
                                         });
    return 1.0 - sum_prob_sq;
}

void DecisionTree::fit(const Matrix<double>& X, const double* y, const std::vector<int>& indices) {
    root_ = build_tree(X, y, indices, 0);
}

std::unique_ptr<Node> DecisionTree::build_tree(const Matrix<double>& X, const double* y, const std::vector<int>& indices, int depth) {
    double predicted_value;
    if (task_type_ == TaskType::REGRESSION) {
        predicted_value = calculate_mean(y, indices);
    } else {
        predicted_value = calculate_mode(y, indices);
    }

    if (depth >= max_depth_ || static_cast<int>(indices.size()) < min_samples_split_) {
        return std::make_unique<Node>(predicted_value);
    }

    bool pure = true;
    if (!indices.empty()) {
        double first = y[indices[0]];
        for (size_t i = 1; i < indices.size(); ++i) {
            if (y[indices[i]] != first) {
                pure = false;
                break;
            }
        }
    }
    if (pure) return std::make_unique<Node>(predicted_value);

    int best_feature = -1;
    double best_threshold = 0.0;
    double best_reduction = -1.0;
    
    double current_impurity;
    if (task_type_ == TaskType::REGRESSION) current_impurity = calculate_variance(y, indices);
    else current_impurity = calculate_gini(y, indices);

    int n_features = X.cols();
    int n_samples = static_cast<int>(indices.size());

    // Sélection aléatoire des features (Feature Bagging au niveau du nœud)
    std::vector<int> feature_indices(n_features);
    std::iota(feature_indices.begin(), feature_indices.end(), 0);

    int m = max_features_;
    if (m <= 0 || m > n_features) {
        if (task_type_ == TaskType::CLASSIFICATION) 
            m = std::max(1, static_cast<int>(std::sqrt(n_features)));
        else 
            m = std::max(1, n_features / 3); // Règle empirique standard pour la régression
    }

    static thread_local std::mt19937 gen(std::random_device{}());
    std::shuffle(feature_indices.begin(), feature_indices.end(), gen);
    feature_indices.resize(m);

    for (int f : feature_indices) {
        struct Sample {
            double val;
            double target;
        };
        std::vector<Sample> samples;
        samples.reserve(n_samples);
        for (int idx : indices) {
            samples.push_back({X(idx, f), y[idx]});
        }
        std::sort(samples.begin(), samples.end(), [](const Sample& a, const Sample& b) {
            return a.val < b.val;
        });

        if (task_type_ == TaskType::REGRESSION) {
            double sum_l = 0, sum_sq_l = 0;
            double sum_r = 0, sum_sq_r = 0;
            for (const auto& s : samples) {
                sum_r += s.target;
                sum_sq_r += s.target * s.target;
            }

            for (int i = 0; i < n_samples - 1; ++i) {
                double t = samples[i].target;
                sum_l += t; sum_sq_l += t * t;
                sum_r -= t; sum_sq_r -= t * t;

                if (samples[i].val == samples[i+1].val) continue;

                int n_l = i + 1;
                int n_r = n_samples - n_l;

                double var_l = std::max(0.0, (sum_sq_l / n_l) - (sum_l / n_l) * (sum_l / n_l));
                double var_r = std::max(0.0, (sum_sq_r / n_r) - (sum_r / n_r) * (sum_r / n_r));
                
                double weighted_impurity = (static_cast<double>(n_l) / n_samples) * var_l + 
                                         (static_cast<double>(n_r) / n_samples) * var_r;
                double reduction = current_impurity - weighted_impurity;

                if (reduction > best_reduction) {
                    best_reduction = reduction;
                    best_feature = f;
                    best_threshold = samples[i].val;
                }
            }
        } else {
            std::map<double, int> counts_l, counts_r;
            double sum_sq_c_l = 0, sum_sq_c_r = 0;

            for (const auto& s : samples) counts_r[s.target]++;
            for (auto const& [val, count] : counts_r) {
                sum_sq_c_r += static_cast<double>(count) * count;
                counts_l[val] = 0; // Pré-initialisation pour éviter les allocations dans la boucle
            }

            for (int i = 0; i < n_samples - 1; ++i) {
                double t = samples[i].target;
                
                sum_sq_c_l -= static_cast<double>(counts_l[t]) * counts_l[t];
                counts_l[t]++;
                sum_sq_c_l += static_cast<double>(counts_l[t]) * counts_l[t];

                sum_sq_c_r -= static_cast<double>(counts_r[t]) * counts_r[t];
                counts_r[t]--;
                sum_sq_c_r += static_cast<double>(counts_r[t]) * counts_r[t];

                if (samples[i].val == samples[i+1].val) continue;

                int n_l = i + 1;
                int n_r = n_samples - n_l;

                double gini_l = 1.0 - (sum_sq_c_l / (static_cast<double>(n_l) * n_l));
                double gini_r = 1.0 - (sum_sq_c_r / (static_cast<double>(n_r) * n_r));

                double weighted_impurity = (static_cast<double>(n_l) / n_samples) * gini_l + 
                                         (static_cast<double>(n_r) / n_samples) * gini_r;
                double reduction = current_impurity - weighted_impurity;

                if (reduction > best_reduction) {
                    best_reduction = reduction;
                    best_feature = f;
                    best_threshold = samples[i].val;
                }
            }
        }
    }

    if (best_feature != -1) {
        auto node = std::make_unique<Node>(best_feature, best_threshold);
        std::vector<int> left_idx, right_idx;
        for (int idx : indices) {
            if (X(idx, best_feature) <= best_threshold) left_idx.push_back(idx);
            else right_idx.push_back(idx);
        }
        node->left = build_tree(X, y, left_idx, depth + 1);
        node->right = build_tree(X, y, right_idx, depth + 1);
        return node;
    }

    return std::make_unique<Node>(predicted_value);
}

double DecisionTree::predict(const Matrix<double>& X, int row_idx) const {
    Node* current = root_.get();
    while (current && (current->left || current->right)) {
        if (X(row_idx, current->feature_index) <= current->threshold) {
            current = current->left.get();
        } else {
            current = current->right.get();
        }
    }
    return current ? current->value : 0.0;
}