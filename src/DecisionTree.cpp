#include "DecisionTree.hpp"
#include <algorithm>
#include <numeric>
#include <set>
#include <map>
#include <limits>
#include <cmath>

// Fonctions utilitaires

static double calculate_mean(const std::vector<double>& y, const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    double sum = 0.0;
    for (int idx : indices) sum += y[idx];
    return sum / indices.size();
}

static double calculate_variance(const std::vector<double>& y, const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    double mean = calculate_mean(y, indices);
    double variance = 0.0;
    for (int idx : indices) {
        double diff = y[idx] - mean;
        variance += diff * diff;
    }
    return variance / indices.size();
}

static double calculate_mode(const std::vector<double>& y, const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    std::map<double, int> counts;
    for (int idx : indices) counts[y[idx]]++;
    
    double mode = 0.0;
    int max_count = -1;
    for (const auto& pair : counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            mode = pair.first;
        }
    }
    return mode;
}

static double calculate_gini(const std::vector<double>& y, const std::vector<int>& indices) {
    if (indices.empty()) return 0.0;
    std::map<double, int> counts;
    for (int idx : indices) counts[y[idx]]++;
    
    double impurity = 1.0;
    double n = static_cast<double>(indices.size());
    for (const auto& pair : counts) {
        double prob = pair.second / n;
        impurity -= prob * prob;
    }
    return impurity;
}

void DecisionTree::fit(const Matrix<double>& X_train, const std::vector<double>& y_train){
    std::vector<int> indices(X_train.rows());
    std::iota(indices.begin(), indices.end(), 0);
    root = build_tree(X_train, y_train, indices, 0);
}

std::unique_ptr<Node> DecisionTree::build_tree(const Matrix<double>& X, const std::vector<double>& y, const std::vector<int>& indices, int depth) {
    double predicted_value;
    if (task_type == TaskType::REGRESSION) {
        predicted_value = calculate_mean(y, indices);
    } else {
        predicted_value = calculate_mode(y, indices);
    }

    if (depth >= max_depth || indices.size() < min_samples_split) {
        return std::make_unique<Node>(predicted_value);
    }

    bool pure = true;
    if (!indices.empty()) {
        double first = y[indices[0]];
        for (int i = 1; i < indices.size(); ++i) {
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
    if (task_type == TaskType::REGRESSION) current_impurity = calculate_variance(y, indices);
    else current_impurity = calculate_gini(y, indices);

    int n_features = X.cols();
    for (int f = 0; f < n_features; ++f) {
        std::set<double> thresholds;
        for (int idx : indices) thresholds.insert(X(idx, f));

        for (double thresh : thresholds) {
            std::vector<int> left_idx, right_idx;
            for (int idx : indices) {
                if (X(idx, f) <= thresh) left_idx.push_back(idx);
                else right_idx.push_back(idx);
            }

            if (left_idx.empty() || right_idx.empty()) continue;

            double imp_left, imp_right;
            if (task_type == TaskType::REGRESSION) {
                imp_left = calculate_variance(y, left_idx);
                imp_right = calculate_variance(y, right_idx);
            } else {
                imp_left = calculate_gini(y, left_idx);
                imp_right = calculate_gini(y, right_idx);
            }

            double w_left = (double)left_idx.size() / indices.size();
            double w_right = (double)right_idx.size() / indices.size();
            double weighted_impurity = w_left * imp_left + w_right * imp_right;
            
            double reduction = current_impurity - weighted_impurity;

            if (reduction > best_reduction) {
                best_reduction = reduction;
                best_feature = f;
                best_threshold = thresh;
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