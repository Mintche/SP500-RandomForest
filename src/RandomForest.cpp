#include "RandomForest.hpp"
#include <random>
#include <algorithm>
#include <map>

RandomForest::RandomForest(int n_trees, int max_depth, int min_samples_split, int n_samples_bootstrap, int max_features, TaskType task_type)
    : n_trees_(n_trees), max_depth_(max_depth), min_samples_split_(min_samples_split),
     n_samples_bootstrap_(n_samples_bootstrap), max_features_(max_features), task_type_(task_type) {
    trees_.reserve(n_trees_);
    for (int i = 0; i < n_trees_; ++i) {
        trees_.push_back(std::make_unique<DecisionTree>(max_depth, min_samples_split, task_type));
    }
}

void RandomForest::fit(const Matrix<double>& X, const std::vector<double>& y) {
    int n_samples = X.rows();
    int n_features = X.cols();
    
    // bootstrap
    int N = (n_samples_bootstrap_ == -1) ? n_samples : n_samples_bootstrap_;

    //features max
    int M;
    if (task_type_ == TaskType::REGRESSION){
        M = (max_features_ == -1) ? n_features/3 : max_features_;
    }
    else{
        M = (max_features_ == -1) ? int(std::sqrt(n_features)) : max_features_;
    }

    Matrix<double> X_boot(N, n_features);
    std::vector<double> y_boot(N);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_sample(0, n_samples - 1);

    for (std::unique_ptr<DecisionTree>& tree : trees_){
        for (int i=0;i<N;i++){
            int i_rd = dis_sample(gen);
            for (int j=0;j<M;j++){
                X_boot(i,j) = X(i_rd,j);
            }
            y_boot[i] = y[i_rd];
        }
        tree->fit(X_boot, y_boot);
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