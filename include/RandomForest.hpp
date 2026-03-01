#pragma once

#include "DecisionTree.hpp"

class RandomForest {
public:
    RandomForest(int n_trees, int max_depth, int min_samples_split, int n_samples_bootstrap = -1, int max_features = -1, TaskType task_type = TaskType::REGRESSION);

    void fit(const Matrix<double>& X, const double* y);
    std::vector<double> predict(const Matrix<double>& X);

private:
    int n_trees_;
    int max_depth_;
    int min_samples_split_;
    int n_samples_bootstrap_;
    int max_features_;
    TaskType task_type_;

    std::vector<std::unique_ptr<DecisionTree>> trees_;
};