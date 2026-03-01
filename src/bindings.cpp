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
        
        .def("fit", [](RandomForest& self, 
                       py::array_t<double, py::array::c_style | py::array::forcecast> X_np, 
                       py::array_t<double, py::array::c_style | py::array::forcecast> y_np) {
            py::buffer_info buf_X = X_np.request();
            py::buffer_info buf_y = y_np.request();

            if (buf_X.ndim != 2) throw std::runtime_error("X doit être en 2D");
            
            // Création d'une vue Matrix sans copie
            Matrix<double> mat_X(buf_X.shape[0], buf_X.shape[1], static_cast<const double*>(buf_X.ptr));
            const double* ptr_y = static_cast<const double*>(buf_y.ptr);

            self.fit(mat_X, ptr_y);
        }, "Entraîner le modèle (Zero-Copy)")

        .def("predict", [](RandomForest& self, py::array_t<double, py::array::c_style | py::array::forcecast> X_np) {
            py::buffer_info buf_X = X_np.request();
            if (buf_X.ndim != 2) throw std::runtime_error("X doit être en 2D");
            
            Matrix<double> mat_X(buf_X.shape[0], buf_X.shape[1], static_cast<const double*>(buf_X.ptr));
            return self.predict(mat_X);
        }, "Prédire les valeurs (Zero-Copy)");
}