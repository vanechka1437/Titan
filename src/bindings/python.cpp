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
        // --------------------------------------------------------------------
        
        // Shape: [num_envs, num_agents] -> uint8_t
        .def_prop_ro("ready_mask", [](UnifiedMemoryArena& a) {
            size_t shape[2] = { static_cast<size_t>(a.num_envs()), static_cast<size_t>(a.num_agents()) };
            return ZeroCopyTensor2D<uint8_t>(a.ready_mask_ptr(), 2, shape, nb::cast(&a, nb::rv_policy::reference));
        })
        
        // Shape: [num_envs, max_actions_per_step, 4] -> int64_t
        .def_prop_ro("actions", [](UnifiedMemoryArena& a) {
            size_t shape[3] = { static_cast<size_t>(a.num_envs()), static_cast<size_t>(a.max_actions_per_step()), 4 };
            return ZeroCopyTensor3D<int64_t>(reinterpret_cast<int64_t*>(a.actions_ptr()), 3, shape, nb::cast(&a, nb::rv_policy::reference));
        })

        // Shape: [num_envs, max_events_per_step, 4] -> int64_t
        // Safely maps the 32-byte MarketDataEvent struct into 4 contiguous int64_t primitives.
        .def_prop_ro("events", [](UnifiedMemoryArena& a) {
            size_t shape[3] = { static_cast<size_t>(a.num_envs()), static_cast<size_t>(a.max_events_per_step()), 4 };
            return ZeroCopyTensor3D<int64_t>(reinterpret_cast<int64_t*>(a.events_ptr()), 3, shape, nb::cast(&a, nb::rv_policy::reference));
        })
        
        // Shape: [num_envs, max_active_orders, 2] -> int64_t
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
    // ========================================================================
    using Sim = BatchSimulator<DEFAULT_OBS_DEPTH>;

    nb::class_<Sim>(m, "Simulator")
        .def(nb::init<UnifiedMemoryArena*, uint32_t, uint32_t>(),
             nb::arg("arena").none(false),
             nb::arg("target_batch_size"),
             nb::arg("num_threads"),
             nb::keep_alive<1, 2>(), 
             "Initializes the asynchronous HFT Discrete Event Simulator.")

        // --------------------------------------------------------------------
        // LIFECYCLE & EXECUTION CONTROLS 
        // --------------------------------------------------------------------
        .def("start", &Sim::start, 
             nb::call_guard<nb::gil_scoped_release>())
             
        .def("stop", &Sim::stop, 
             nb::call_guard<nb::gil_scoped_release>())
        
        .def("resume_batch", &Sim::resume_batch,
             nb::call_guard<nb::gil_scoped_release>())
        
        .def("wait_for_batch", &Sim::wait_for_batch,
             nb::call_guard<nb::gil_scoped_release>())

        // --------------------------------------------------------------------
        // HARDWARE & NETWORK LATENCY
        // --------------------------------------------------------------------
        .def("set_agent_latencies", &Sim::set_agent_latencies,
             nb::arg("agent_id"), 
             nb::arg("ingress_ns"), 
             nb::arg("egress_ns"), 
             nb::arg("compute_ns"),
             nb::call_guard<nb::gil_scoped_release>(),
             "Sets the physical network and compute latencies for a specific agent across all environments.")
             
        // --------------------------------------------------------------------
        // DYNAMIC STATE EXPORTS
        // --------------------------------------------------------------------
        
        // Shape: [num_envs] -> uint64_t
        // Dynamically copies the environment clock since it is interleaved memory in C++
        .def("get_current_times", [](Sim& sim, uint32_t num_envs) {
            uint64_t* data = new uint64_t[num_envs];
            for (uint32_t i = 0; i < num_envs; ++i) {
                data[i] = sim.get_env(i).current_time;
            }
            size_t shape[1] = { num_envs };
            nb::capsule owner(data, [](void *p) noexcept { delete[] (uint64_t *) p; });
            return ZeroCopyTensor1D<uint64_t>(data, 1, shape, owner);
        }, nb::arg("num_envs"), "Returns a dynamically allocated 1D tensor of current environment timestamps.")
        
        // --------------------------------------------------------------------
        // VECTORIZED AUTO-RESETS
        // --------------------------------------------------------------------
        .def("reset_all", &Sim::reset_all, 
             nb::call_guard<nb::gil_scoped_release>())
        
        .def("reset", &Sim::reset, 
             nb::arg("env_indices"), 
             nb::call_guard<nb::gil_scoped_release>())

        .def("inject_order", [](BatchSimulator<DEFAULT_OBS_DEPTH>& self, uint32_t env_idx, int32_t price, int64_t qty, int side, uint16_t agent_id) {
            auto& env = self.get_env(env_idx);
            titan::core::ActionPayload ap{};
            ap.action_type = 0; // LIMIT
            ap.price = static_cast<uint32_t>(price);
            ap.qty = qty;
            ap.side = static_cast<uint8_t>(side);
            ap.agent_id = agent_id;
            
            self.get_scheduler(env_idx).push(
                titan::core::ScheduledEvent::make_order_arrival(env.current_time, agent_id, ap)
            );
        }, nb::arg("env_idx"), nb::arg("price"), nb::arg("qty"), nb::arg("side"), nb::arg("agent_id"));
}