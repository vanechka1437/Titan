#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include "titan/core/batch_simulator.hpp"

namespace nb = nanobind;
using namespace titan::core;

// ============================================================================
// ZERO-COPY DLPack CONFIGURATION
// Specifies explicit C-contiguous memory layout for PyTorch and NumPy interoperability.
// ============================================================================
template <typename T>
using ZeroCopyTensor1D = nb::ndarray<T, nb::pytorch, nb::numpy, nb::shape<nb::any>, nb::c_contig>;

template <typename T>
using ZeroCopyTensor2D = nb::ndarray<T, nb::pytorch, nb::numpy, nb::shape<nb::any, nb::any>, nb::c_contig>;

template <typename T>
using ZeroCopyTensor3D = nb::ndarray<T, nb::pytorch, nb::numpy, nb::shape<nb::any, nb::any, nb::any>, nb::c_contig>;

// ============================================================================
// PYTHON BINDINGS MODULE
// ============================================================================
NB_MODULE(titan_core, m) {
    m.doc() = "Titan HFT Core - Zero-Copy PyTorch RL Environment";

    // Expose the Simulator (Defaulting to ObsDepth = 20)
    using Sim = BatchSimulator<20>;

    nb::class_<Sim>(m, "Simulator")
        // --------------------------------------------------------------------
        // 1. CONSTRUCTOR
        // --------------------------------------------------------------------
        .def(nb::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
             nb::arg("num_envs"),
             nb::arg("num_agents_per_env"),
             nb::arg("max_actions_per_step") = 10,
             nb::arg("max_active_orders") = 10000,
             "Initialize the massively parallel HFT simulator.")

        // --------------------------------------------------------------------
        // 2. CORE EXECUTION
        // --------------------------------------------------------------------
        .def("resume_batch", &Sim::resume_batch,
             nb::call_guard<nb::gil_scoped_release>(), 
             "Resumes the physics engine until agents need to wake up for decisions.")
        
        .def("reset_all", &Sim::reset_all,
             nb::call_guard<nb::gil_scoped_release>(),
             "Hard reset of all memory pools and environments.")
        
        .def("reset", &Sim::reset,
             nb::arg("env_indices"),
             nb::call_guard<nb::gil_scoped_release>(),
             "Targeted reset of specific environments (Vectorized Auto-Reset).")

        // --------------------------------------------------------------------
        // 3. ZERO-COPY DLPACK EXPORTS
        // nb::cast with rv_policy::reference anchors the DLPack capsule to the
        // Simulator instance, preventing Garbage Collection while PyTorch holds it.
        // --------------------------------------------------------------------
        
        // READY MASK [num_envs, num_agents] -> uint8
        .def_prop_ro("ready_mask", [](Sim& this_) {
            size_t shape[2] = { static_cast<size_t>(this_.num_envs()), 
                                static_cast<size_t>(this_.num_agents()) };
            return ZeroCopyTensor2D<uint8_t>(
                this_.arena()->ready_mask_ptr(), 
                2, shape, 
                nb::cast(&this_, nb::rv_policy::reference)
            );
        })

        // ACTIONS TENSOR [num_envs, max_actions, 4] -> int64
        // Safely exports the 32-byte ActionPayload as 4 contiguous int64 elements.
        // Prevents PyTorch non-contiguous slice crashes during bitwise operations.
        .def_prop_ro("actions", [](Sim& this_) {
            size_t shape[3] = { static_cast<size_t>(this_.num_envs()), 
                                static_cast<size_t>(this_.arena()->max_actions_per_step()), 
                                4 };
            return ZeroCopyTensor3D<int64_t>(
                reinterpret_cast<int64_t*>(this_.arena()->actions_ptr()), 
                3, shape, 
                nb::cast(&this_, nb::rv_policy::reference)
            );
        })

        // ACTIVE ORDERS [num_envs, max_active_orders, 2] -> int64
        // Exports the 16-byte ActiveOrderRecord as 2 contiguous int64 elements.
        .def_prop_ro("active_orders", [](Sim& this_) {
            size_t shape[3] = { static_cast<size_t>(this_.num_envs()), 
                                static_cast<size_t>(this_.arena()->max_active_orders()), 
                                2 };
            return ZeroCopyTensor3D<int64_t>(
                reinterpret_cast<int64_t*>(this_.arena()->active_orders_ptr()), 
                3, shape, 
                nb::cast(&this_, nb::rv_policy::reference)
            );
        })

        // EVENT CURSORS [num_envs] -> int64
        // Monotonic ring-buffer cursors for lock-free event history reading in PyTorch
        .def_prop_ro("event_cursors", [](Sim& this_) {
            size_t shape[1] = { static_cast<size_t>(this_.num_envs()) };
            return ZeroCopyTensor1D<int64_t>(
                reinterpret_cast<int64_t*>(this_.arena()->event_cursors_ptr()), 
                1, shape, 
                nb::cast(&this_, nb::rv_policy::reference)
            );
        })

        // --------------------------------------------------------------------
        // 4. UTILITY PROPERTIES
        // --------------------------------------------------------------------
        .def_prop_ro("num_envs", &Sim::num_envs)
        .def_prop_ro("num_agents", &Sim::num_agents)
        .def_prop_ro("ready_count", [](const Sim& this_) {
            return this_.ready_count_.load(std::memory_order_relaxed);
        });
}