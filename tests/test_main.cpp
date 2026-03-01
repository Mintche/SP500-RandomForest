#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "RandomForest.hpp"

void test_regression() {
    std::cout << "Running Regression Test..." << std::endl;
    
    // Création d'un dataset simple : y = x
    int n_samples = 20;
    Matrix<double> X(n_samples, 1);
    std::vector<double> y(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        X(i, 0) = static_cast<double>(i);
        y[i] = static_cast<double>(i);
    }

    // Configuration : 5 arbres, profondeur 5
    RandomForest rf(5, 5, 2, -1, -1, TaskType::REGRESSION);
    rf.fit(X, y);

    // Test sur une valeur connue
    Matrix<double> X_test(1, 1);
    X_test(0, 0) = 10.0;
    std::vector<double> pred = rf.predict(X_test);

    std::cout << "Pred for 10.0: " << pred[0] << " (Expected ~10.0)" << std::endl;
    assert(std::abs(pred[0] - 10.0) < 2.0); 
    std::cout << "Regression Test Passed!" << std::endl;
}

void test_classification() {
    std::cout << "\nRunning Classification Test..." << std::endl;

    // Dataset : si x < 5 alors 0, sinon 1
    int n_samples = 20;
    Matrix<double> X(n_samples, 1);
    std::vector<double> y(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        X(i, 0) = static_cast<double>(i);
        y[i] = (i < 10) ? 0.0 : 1.0;
    }

    RandomForest rf(10, 3, 2, -1, -1, TaskType::CLASSIFICATION);
    rf.fit(X, y);

    Matrix<double> X_test(2, 1);
    X_test(0, 0) = 2.0;  // Devrait être 0
    X_test(1, 0) = 18.0; // Devrait être 1
    
    std::vector<double> preds = rf.predict(X_test);

    std::cout << "Pred for 2.0: " << preds[0] << " (Expected 0.0)" << std::endl;
    std::cout << "Pred for 18.0: " << preds[1] << " (Expected 1.0)" << std::endl;

    assert(preds[0] == 0.0);
    assert(preds[1] == 1.0);
    std::cout << "Classification Test Passed!" << std::endl;
}

int main() {
    try {
        test_regression();
        test_classification();
        std::cout << "\nAll C++ tests passed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}