#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Pour la conversion automatique std::vector <-> list/numpy
#include <pybind11/numpy.h> // Pour supporter les numpy arrays
#include "RandomForest.hpp"

namespace py = pybind11;

PYBIND11_MODULE(RandomForestcpp, m) {
    m.doc() = "Module C++ RandomForest pour la prédiction SP500";

    py::enum_<TaskType>(m, "TaskType")
        .value("REGRESSION", TaskType::REGRESSION)
        .value("CLASSIFICATION", TaskType::CLASSIFICATION)
        .export_values();

    py::class_<RandomForest>(m, "RandomForest")
        .def(py::init<int, int, int, int, int, TaskType>(), 
             py::arg("n_trees") = 100, 
             py::arg("max_depth") = 10, 
             py::arg("min_samples_split") = 2,
             py::arg("n_samples_bootstrap") = -1,
             py::arg("max_features") = -1,
             py::arg("task_type") = TaskType::REGRESSION)
        
        // Surcharge pour accepter des numpy arrays directement
        .def("fit", [](RandomForest& self, py::array_t<double> X_np, py::array_t<double> y_np) {
            py::buffer_info buf_X = X_np.request();
            py::buffer_info buf_y = y_np.request();

            if (buf_X.ndim != 2) throw std::runtime_error("X doit être en 2D");
            
            // Conversion vers Matrix<double>
            // On suppose que Matrix a un constructeur (rows, cols) et l'opérateur (i, j)
            Matrix<double> mat_X(buf_X.shape[0], buf_X.shape[1]);
            double* ptr_X = static_cast<double*>(buf_X.ptr);
            for (size_t i = 0; i < buf_X.shape[0]; i++) {
                for (size_t j = 0; j < buf_X.shape[1]; j++) {
                    mat_X(i, j) = ptr_X[i * buf_X.shape[1] + j];
                }
            }

            // Conversion vers std::vector<double> pour y
            std::vector<double> vec_y(buf_y.size);
            double* ptr_y = static_cast<double*>(buf_y.ptr);
            for (size_t i = 0; i < buf_y.size; i++) vec_y[i] = ptr_y[i];

            self.fit(mat_X, vec_y);
        }, "Entraîner le modèle avec des numpy arrays")

        .def("predict", [](RandomForest& self, py::array_t<double> X_np) {
            py::buffer_info buf_X = X_np.request();
            if (buf_X.ndim != 2) throw std::runtime_error("X doit être en 2D");

            Matrix<double> mat_X(buf_X.shape[0], buf_X.shape[1]);
            double* ptr_X = static_cast<double*>(buf_X.ptr);
            for (size_t i = 0; i < buf_X.shape[0]; i++) {
                for (size_t j = 0; j < buf_X.shape[1]; j++) {
                    mat_X(i, j) = ptr_X[i * buf_X.shape[1] + j];
                }
            }

            return self.predict(mat_X);
        }, "Prédire les valeurs avec des numpy arrays");
}