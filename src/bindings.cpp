#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Pour la conversion automatique std::vector <-> list/numpy
#include "RandomForest.hpp"

namespace py = pybind11;

PYBIND11_MODULE(sp500_rf_cpp, m) {
    m.doc() = "Module C++ RandomForest pour la prédiction SP500";

    py::class_<RandomForest>(m, "RandomForest")
        .def(py::init<int, int, int>(), 
             py::arg("n_trees") = 100, 
             py::arg("max_depth") = 10, 
             py::arg("min_samples_split") = 2)
        .def("fit", &RandomForest::fit, "Entraîner le modèle")
        .def("predict", &RandomForest::predict, "Prédire les valeurs");
}