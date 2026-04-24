#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include "titan/core/batch_simulator.hpp"
#include "titan/core/memory.hpp"
#include "titan/core/types.hpp"

namespace nb = nanobind;
using namespace titan::core;

// ============================================================================
// ZERO-COPY DLPACK CONFIGURATION
// Specifies explicit C-contiguous memory layout for PyTorch / NumPy interoperability.
// Enables O(1) instantaneous memory mapping without deep copies or serialization.
// Using nb::ndim<N> guarantees safe dimension inference in nanobind v2.0+.
// ============================================================================
template <typename T>
using ZeroCopyTensor1D = nb::ndarray<T, nb::pytorch, nb::numpy, nb::ndim<1>, nb::c_contig>;

template <typename T>
using ZeroCopyTensor2D = nb::ndarray<T, nb::pytorch, nb::numpy, nb::ndim<2>, nb::c_contig>;

template <typename T>
using ZeroCopyTensor3D = nb::ndarray<T, nb::pytorch, nb::numpy, nb::ndim<3>, nb::c_contig>;

NB_MODULE(titan_core, m) {
    m.doc() = "Titan HFT Core - Zero-Copy PyTorch RL Environment";

    // ========================================================================
    // 1. UNIFIED MEMORY ARENA (The Master Data Structure)
    // The Python layer explicitly instantiates and owns the master memory block.
    // This perfectly mirrors the C++ Data-Oriented architecture and ensures 
    // the Python Garbage Collector preserves the memory while PyTorch holds tensors.
    // ========================================================================
    nb::class_<UnifiedMemoryArena>(m, "MemoryArena")
        .def(nb::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, std::size_t>(),
             nb::arg("num_envs"),
             nb::arg("num_agents"),
             nb::arg("max_orders_per_env") = 4096,
             nb::arg("max_actions_per_step") = 16,
             nb::arg("max_events_per_step") = 256,
             nb::arg("max_orders_per_agent") = 128,
             nb::arg("obs_depth") = 20,
             nb::arg("linear_bytes") = 256 * 1024 * 1024,
             "Allocates OS-pinned, zero-copy contiguous memory for the entire parallel simulation.")
             
        // --------------------------------------------------------------------
        // ZERO-COPY TENSOR EXPORTS (Anchored to the Arena's lifetime)
        // nb::cast with rv_policy::reference ensures PyTorch references 
        // the raw C++ memory securely without invoking costly array copies.
        // --------------------------------------------------------------------
        
        // Shape: [num_envs, num_agents] -> uint8_t
        .def_prop_ro("ready_mask", [](UnifiedMemoryArena& a) {
            size_t shape[2] = { static_cast<size_t>(a.num_envs()), static_cast<size_t>(a.num_agents()) };
            return ZeroCopyTensor2D<uint8_t>(a.ready_mask_ptr(), 2, shape, nb::cast(&a, nb::rv_policy::reference));
        })
        
        // Shape: [num_envs, max_actions_per_step, 4] -> int64_t
        // Safely maps the 32-byte ActionPayload struct into 4 contiguous int64_t primitives.
        .def_prop_ro("actions", [](UnifiedMemoryArena& a) {
            size_t shape[3] = { static_cast<size_t>(a.num_envs()), static_cast<size_t>(a.max_actions_per_step()), 4 };
            return ZeroCopyTensor3D<int64_t>(reinterpret_cast<int64_t*>(a.actions_ptr()), 3, shape, nb::cast(&a, nb::rv_policy::reference));
        })
        
        // Shape: [num_envs, max_active_orders, 2] -> int64_t
        // Maps the 16-byte ActiveOrderRecord struct into 2 contiguous int64_t primitives.
        .def_prop_ro("active_orders", [](UnifiedMemoryArena& a) {
            size_t shape[3] = { static_cast<size_t>(a.num_envs()), static_cast<size_t>(a.max_active_orders()), 2 };
            return ZeroCopyTensor3D<int64_t>(reinterpret_cast<int64_t*>(a.active_orders_ptr()), 3, shape, nb::cast(&a, nb::rv_policy::reference));
        })
        
        // Shape: [num_envs] -> int64_t
        .def_prop_ro("event_cursors", [](UnifiedMemoryArena& a) {
            size_t shape[1] = { static_cast<size_t>(a.num_envs()) };
            return ZeroCopyTensor1D<int64_t>(reinterpret_cast<int64_t*>(a.event_cursors_ptr()), 1, shape, nb::cast(&a, nb::rv_policy::reference));
        })
        
        // Shape: [num_envs, num_agents, obs_depth * 4] -> float32
        .def_prop_ro("lob", [](UnifiedMemoryArena& a) {
            size_t shape[3] = { static_cast<size_t>(a.num_envs()), static_cast<size_t>(a.num_agents()), static_cast<size_t>(a.obs_depth() * 4) };
            return ZeroCopyTensor3D<float>(a.lob_ptr(), 3, shape, nb::cast(&a, nb::rv_policy::reference));
        })
        
        // Shape: [num_envs, num_agents] -> float32
        .def_prop_ro("cash", [](UnifiedMemoryArena& a) {
            size_t shape[2] = { static_cast<size_t>(a.num_envs()), static_cast<size_t>(a.num_agents()) };
            return ZeroCopyTensor2D<float>(a.cash_ptr(), 2, shape, nb::cast(&a, nb::rv_policy::reference));
        })
        
        // Shape: [num_envs, num_agents] -> float32
        .def_prop_ro("inventory", [](UnifiedMemoryArena& a) {
            size_t shape[2] = { static_cast<size_t>(a.num_envs()), static_cast<size_t>(a.num_agents()) };
            return ZeroCopyTensor2D<float>(a.inventory_ptr(), 2, shape, nb::cast(&a, nb::rv_policy::reference));
        })

        // Structural Properties
        .def_prop_ro("num_envs", &UnifiedMemoryArena::num_envs)
        .def_prop_ro("num_agents", &UnifiedMemoryArena::num_agents);

    // ========================================================================
    // 2. BATCH SIMULATOR (The Execution Engine)
    // Runs the multi-threaded physics calculations over the Arena's memory.
    // ========================================================================
    using Sim = BatchSimulator<DEFAULT_OBS_DEPTH>;

    nb::class_<Sim>(m, "Simulator")
        .def(nb::init<UnifiedMemoryArena*, uint32_t, uint32_t>(),
             nb::arg("arena").none(false),
             nb::arg("target_batch_size"),
             nb::arg("num_threads"),
             // CRITICAL SAFETY MECHANISM: keep_alive<1, 2>() rigidly links the 
             // Simulator's lifecycle (arg 1) to the Arena (arg 2). This ensures 
             // Python cannot garbage collect the memory while threads are active.
             nb::keep_alive<1, 2>(), 
             "Initializes the asynchronous HFT Discrete Event Simulator.")

        // --------------------------------------------------------------------
        // LIFECYCLE & EXECUTION CONTROLS 
        // GIL is strictly dropped (gil_scoped_release) during execution to 
        // permit true concurrent scaling between Python backprop and C++ physics.
        // --------------------------------------------------------------------
        .def("start", &Sim::start, 
             nb::call_guard<nb::gil_scoped_release>(),
             "Ignites the C++ worker thread pool.")
             
        .def("stop", &Sim::stop, 
             nb::call_guard<nb::gil_scoped_release>(),
             "Safely joins the background thread pool and halts the engine.")
        
        .def("resume_batch", &Sim::resume_batch,
             nb::call_guard<nb::gil_scoped_release>(), 
             "Notifies the C++ engine to ingest new RL actions from memory and resume physics.")
        
        .def("wait_for_batch", &Sim::wait_for_batch,
             nb::call_guard<nb::gil_scoped_release>(), 
             "Blocks the Python execution thread until 'target_batch_size' agents require inference.")
        
        // --------------------------------------------------------------------
        // VECTORIZED AUTO-RESETS
        // --------------------------------------------------------------------
        .def("reset_all", &Sim::reset_all, 
             nb::call_guard<nb::gil_scoped_release>(),
             "Global flush: instantaneously wipes all memory boundaries to step 0.")
        
        .def("reset", &Sim::reset, 
             nb::arg("env_indices"), 
             nb::call_guard<nb::gil_scoped_release>(),
             "Targeted reset: Wipes states exclusively for terminated environments (O(1) episode reset).");
}