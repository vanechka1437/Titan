#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <type_traits>
#include <vector>

// External Dependencies
#include <boost/heap/d_ary_heap.hpp>
#include <boost/heap/pairing_heap.hpp>

#include "titan/core/scheduler.hpp"

using namespace titan::core;

// ============================================================================
// 1. TYPE TRAITS & SFINAE DETECTORS
// ============================================================================

// Detector for Titan's raw 16-byte heap push: push(uint64_t, uint32_t)
template <typename T, typename = void>
struct has_titan_raw_push : std::false_type {};

template <typename T>
struct has_titan_raw_push<T, std::void_t<decltype(std::declval<T>().push(std::declval<uint64_t>(), std::declval<uint32_t>()))>>
    : std::true_type {};

// ============================================================================
// 2. DISPATCHERS (Push & Factory for 16-byte raw heaps)
// ============================================================================

template <typename Scheduler>
void do_push(Scheduler& heap, uint64_t time, uint32_t idx) {
    if constexpr (has_titan_raw_push<Scheduler>::value) {
        // Titan FastDAryHeap
        heap.push(time, idx);
    } else {
        // std::priority_queue and Boost heaps
        heap.push({time, idx, 0});
    }
}

template <typename Scheduler>
Scheduler create_scheduler(size_t capacity) {
    if constexpr (has_titan_raw_push<Scheduler>::value) {
        // Custom Titan heaps require pre-allocation to guarantee no reallocs
        return Scheduler(capacity);
    } else {
        // Boost and STL containers use dynamic growth
        return Scheduler();
    }
}

// ============================================================================
// 3. COMPARATORS & INFRASTRUCTURE
// ============================================================================

// Key comparator for 16-byte struct
struct MinHeapCompare {
    inline bool operator()(const HeapNode& a, const HeapNode& b) const noexcept {
        if (a.arrival_time != b.arrival_time) return a.arrival_time > b.arrival_time;
        return a.sequence_id > b.sequence_id;
    }
};

using StdBinaryHeap = std::priority_queue<HeapNode, std::vector<HeapNode>, MinHeapCompare>;
using BoostPairingHeap = boost::heap::pairing_heap<HeapNode, boost::heap::compare<MinHeapCompare>>;

template <uint32_t D>
using BoostDAryHeap = boost::heap::d_ary_heap<HeapNode, boost::heap::arity<D>, boost::heap::compare<MinHeapCompare>>;

// ============================================================================
// 4. THE ULTIMATE FAT-EVENT COMPARISON (16-byte keys vs 48-byte structs)
// ============================================================================

// Comparator for the full 48-byte struct directly inside std::priority_queue
struct FatEventCompare {
    inline bool operator()(const ScheduledEvent& a, const ScheduledEvent& b) const noexcept {
        return a.time > b.time;
    }
};

using StdFatQueue = std::priority_queue<ScheduledEvent, std::vector<ScheduledEvent>, FatEventCompare>;

template <typename Scheduler>
void do_fat_push(Scheduler& sched, uint64_t time, uint32_t target) {
    sched.push(ScheduledEvent::make_agent_wakeup(time, target));
}

// ============================================================================
// 5. BENCHMARK SCENARIOS (Raw Keys)
// ============================================================================

template <class Scheduler>
void BM_Burst(benchmark::State& state) {
    const size_t batch_size = state.range(0);
    std::mt19937_64 gen(42);
    std::vector<uint64_t> times(batch_size);
    for (auto& t : times) t = gen();

    for (auto _ : state) {
        auto heap = create_scheduler<Scheduler>(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            do_push(heap, times[i], (uint32_t)i);
        }
        while (!heap.empty()) {
            benchmark::DoNotOptimize(heap.top());
            heap.pop();
        }
    }
}

template <class Scheduler>
void BM_Churn(benchmark::State& state) {
    const size_t queue_depth = state.range(0);
    std::mt19937_64 gen(1337);
    auto heap = create_scheduler<Scheduler>(queue_depth + 1);

    for (size_t i = 0; i < queue_depth; ++i) {
        do_push(heap, gen() % 1000000, (uint32_t)i);
    }

    for (auto _ : state) {
        auto current = heap.top();
        heap.pop();
        benchmark::DoNotOptimize(current);
        do_push(heap, current.arrival_time + (gen() % 5000), current.payload_idx);
    }
}

// ============================================================================
// 6. BENCHMARK SCENARIOS (Fat Events: Cache Thrashing Test)
// ============================================================================

template <class Scheduler>
void BM_FatChurn(benchmark::State& state) {
    const size_t queue_depth = state.range(0);
    std::mt19937_64 gen(1337);
    
    // Allocate Memory
    alignas(64) std::byte buffer[sizeof(Scheduler)];
    Scheduler* sched_ptr;
    
    if constexpr (std::is_same_v<Scheduler, FastScheduler>) {
        sched_ptr = new (buffer) Scheduler(queue_depth + 1);
    } else {
        sched_ptr = new (buffer) Scheduler();
    }
    
    Scheduler& sched = *sched_ptr;

    // Fill Queue
    for (size_t i = 0; i < queue_depth; ++i) {
        do_fat_push(sched, gen() % 1000000, (uint32_t)i);
    }

    // Steady State Churn
    for (auto _ : state) {
        auto current = sched.top();
        sched.pop();
        benchmark::DoNotOptimize(current);
        uint64_t next_time = current.time + (gen() % 5000);
        do_fat_push(sched, next_time, current.target_id);
    }
    
    sched_ptr->~Scheduler();
}

// ============================================================================
// 7. REGISTRATION MATRIX
// ============================================================================

#define REGISTER_RAW(CLASS, NAME)                                                   \
    BENCHMARK(BM_Churn<CLASS>)->Name("RawChurn_" NAME)->RangeMultiplier(8)->Range(128, 1024 * 1024); \
    BENCHMARK(BM_Burst<CLASS>)->Name("RawBurst_" NAME)->RangeMultiplier(8)->Range(128, 1024 * 1024);

#define REGISTER_FAT(CLASS, NAME)                                                   \
    BENCHMARK(BM_FatChurn<CLASS>)->Name("FatChurn_" NAME)->RangeMultiplier(8)->Range(128, 1024 * 1024);

// ----------------------------------------------------------------------------
// GROUP 1: Raw 16-byte Key Performance (CPU Sorting Efficiency)
// ----------------------------------------------------------------------------

// C++ STL Baseline
REGISTER_RAW(StdBinaryHeap, "Std_Binary_PQ");

// Boost Library Challengers
REGISTER_RAW(BoostPairingHeap, "Boost_Pairing");
REGISTER_RAW(BoostDAryHeap<4>, "Boost_D4");
REGISTER_RAW(BoostDAryHeap<8>, "Boost_D8");

// Titan HFT Custom Heaps
REGISTER_RAW(FastDAryHeap<2>, "Titan_D2");
REGISTER_RAW(FastDAryHeap<4>, "Titan_D4");
REGISTER_RAW(FastDAryHeap<8>, "Titan_D8");
REGISTER_RAW(FastDAryHeap<16>, "Titan_D16");

// ----------------------------------------------------------------------------
// GROUP 2: The Fat Payload Cache Test (16-byte + Pool vs 48-byte Direct Sort)
// ----------------------------------------------------------------------------

REGISTER_FAT(StdFatQueue, "Std_Fat_48Byte_PQ");
REGISTER_FAT(FastScheduler, "Titan_Separated_Pool");

// ============================================================================
// ENTRY POINT
// ============================================================================
int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    std::cout << "-----------------------------------------------------------\n";
    std::cout << "TITAN SCHEDULER CROSS-ARITY HARDWARE SEARCH\n";
    std::cout << "Testing optimal D-factor for L1/L2 cache prefetching.\n";
    std::cout << "Testing Fat-Event cache fragmentation.\n";
    std::cout << "-----------------------------------------------------------\n";
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}