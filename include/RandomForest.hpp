#pragma once

#include "DecisionTree.hpp"
#include <vector>
#include <memory>

class RandomForest {
public:
    // Constructeur de la forêt
    RandomForest(int n_trees, int max_depth, int min_samples_split);
    ~RandomForest() = default;

    // Méthode d'entraînement (prend une matrice 2D et un vecteur cible)
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);

    // Méthode de prédiction (retourne les prédictions pour plusieurs entrées)
    std::vector<double> predict(const std::vector<std::vector<double>>& X);

private:
    int n_trees_;
    int max_depth_;
    int min_samples_split_;

    // Stockage des arbres
    std::vector<std::unique_ptr<DecisionTree>> trees_;
};