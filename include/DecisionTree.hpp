#ifndef DECISION_TREE_HPP
#define DECISION_TREE_HPP

#include <vector>
#include <memory>

template <typename T>
class Matrix {
    private :
        size_t n_rows;
        size_t n_cols;
        std::vector<T> data;
    
};

class DecisionTree {
public:
    // Constructeur avec hyperparamètres de base
    DecisionTree(int max_depth = 10, int min_samples_split = 2);
    ~DecisionTree() = default;

    // Entraînement sur un sous-ensemble de données
    void fit(const std::vector<std::vector<double>>& features, const std::vector<double>& targets);

    // Prédiction pour une seule entrée
    double predict(const std::vector<double>& features) const;

private:
    int max_depth_;
    int min_samples_split_;
    
    // Structure interne du noeud (à définir dans le cpp ou struct privée)
};

#endif