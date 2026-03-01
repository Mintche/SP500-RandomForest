#include "RandomForest.hpp"
#include <random>
#include <algorithm>
#include <map>
#include <omp.h>

RandomForest::RandomForest(int n_trees, int max_depth, int min_samples_split, int n_samples_bootstrap, int max_features, TaskType task_type)
    : n_trees_(n_trees), max_depth_(max_depth), min_samples_split_(min_samples_split),
     n_samples_bootstrap_(n_samples_bootstrap), max_features_(max_features), task_type_(task_type) {
    trees_.reserve(n_trees_);
    for (int i = 0; i < n_trees_; ++i) {
        trees_.push_back(std::make_unique<DecisionTree>(max_depth, min_samples_split, max_features, task_type));
    }
}

void RandomForest::fit(const Matrix<double>& X, const double* y) {
    int n_samples = X.rows();
    
    // Taille du bootstrap
    int N = (n_samples_bootstrap_ == -1) ? n_samples : n_samples_bootstrap_;

    #pragma omp parallel for
    for (int i = 0; i < n_trees_; ++i) {
        // Chaque thread a besoin de son propre générateur de nombres aléatoires
        std::random_device rd;
        std::mt19937 gen(rd() ^ i); 
        std::uniform_int_distribution<> dis_sample(0, n_samples - 1);

        std::vector<int> boot_indices(N);
        for (int j = 0; j < N; ++j) {
            boot_indices[j] = dis_sample(gen);
        }

        trees_[i]->fit(X, y, boot_indices);
    }
}

std::vector<double> RandomForest::predict(const Matrix<double>& X) {
    int n_samples = X.rows();
    std::vector<double> y_pred(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        if (task_type_ == TaskType::REGRESSION) {
            double sum = 0.0;
            for (const auto& tree : trees_) {
                sum += tree->predict(X, i);
            }
            y_pred[i] = sum / n_trees_;
        } else {
            std::map<double, int> counts;
            for (const auto& tree : trees_) {
                counts[tree->predict(X, i)]++;
            }

            double mode_val = 0.0;
            int max_count = -1;
            for (const auto& pair : counts) {
                if (pair.second > max_count) {
                    max_count = pair.second;
                    mode_val = pair.first;
                }
            }
            y_pred[i] = mode_val;
        }
    }
    return y_pred;
}