#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <vector>
#include <memory>

enum class TaskType {
    REGRESSION,
    CLASSIFICATION
};

template <typename T>
class Matrix {
    private :
        int n_rows;
        int n_cols;
        std::vector<T> data;
    public :
        Matrix(int rows, int cols) : n_rows(rows), n_cols(cols), data(rows * cols) {}
        ~Matrix() {};
        inline T& operator()(int row, int col) {
        return data[row * n_cols + col];
        }
        inline T operator()(int row, int col) const {
        return data[row * n_cols + col];
        }
        int rows() const { return n_rows; }
        int cols() const { return n_cols; }
};

struct Node {
    int feature_index;
    double threshold;
    double value;
    
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;

    Node(int feature, double thresh) : feature_index(feature), threshold(thresh), value(0.0) {};
    Node(double val) : feature_index(-1), threshold(0.0), value(val) {};

    bool is_leaf() const {
        return left == nullptr && right == nullptr;
    };
};

class DecisionTree {
    private:
        int max_depth_;
        int min_samples_split_;
        TaskType task_type_;
        std::unique_ptr<Node> root_;
        std::unique_ptr<Node> build_tree(const Matrix<double>& X, const std::vector<double>& y, const std::vector<int>& indices, int depth);
    public:
        DecisionTree(int max_depth = 10, int min_samples_split = 2, TaskType task_type = TaskType::REGRESSION) : max_depth_(max_depth), min_samples_split_(min_samples_split), task_type_(task_type), root_(nullptr) {}
        ~DecisionTree() {};

        void fit(const Matrix<double>& X_train, const std::vector<double>& y_train);
        double predict(const Matrix<double>& X, int row_idx) const;
};

#endif