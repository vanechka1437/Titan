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

// Detector for Titan-style push: push(uint64_t, uint32_t)
template <typename T, typename = void>
struct has_titan_push : std::false_type {};

template <typename T>
struct has_titan_push<T,
                      std::void_t<decltype(std::declval<T>().push(std::declval<uint64_t>(), std::declval<uint32_t>()))>>
    : std::true_type {};

// Detector for Boost D-Ary heaps (to differentiate from STL/Pairing)
template <typename T>
struct is_boost_dary : std::false_type {};

template <uint32_t D, typename T, typename C>
struct is_boost_dary<boost::heap::d_ary_heap<T, boost::heap::arity<D>, boost::heap::compare<C>>> : std::true_type {};

// ============================================================================
// 2. DISPATCHERS (Push & Factory)
// ============================================================================

template <typename Scheduler>
void do_push(Scheduler& heap, uint64_t time, uint32_t idx) {
    if constexpr (has_titan_push<Scheduler>::value) {
        // Our custom heaps and VectorSort
        heap.push(time, idx);
    } else {
        // std::priority_queue and Boost heaps
        heap.push({time, idx, 0});
    }
}

template <typename Scheduler>
Scheduler create_scheduler(size_t capacity) {
    if constexpr (has_titan_push<Scheduler>::value) {
        // Custom heaps require pre-allocation
        return Scheduler(capacity);
    } else {
        // Boost and STL containers use dynamic growth
        return Scheduler();
    }
}

// ============================================================================
// 3. COMPARATORS & INFRASTRUCTURE
// ============================================================================

struct MinHeapCompare {
    inline bool operator()(const HeapNode& a, const HeapNode& b) const noexcept {
        return a.arrival_time > b.arrival_time;
    }
};

using StdBinaryHeap = std::priority_queue<HeapNode, std::vector<HeapNode>, MinHeapCompare>;
using BoostPairingHeap = boost::heap::pairing_heap<HeapNode, boost::heap::compare<MinHeapCompare>>;

template <uint32_t D>
using BoostDAryHeap = boost::heap::d_ary_heap<HeapNode, boost::heap::arity<D>, boost::heap::compare<MinHeapCompare>>;

// Challenger: Simple vector + postponed sort
class VectorSortScheduler {
private:
    std::vector<HeapNode> data_;
    bool is_sorted_{true};

public:
    explicit VectorSortScheduler(size_t cap) { data_.reserve(cap); }
    inline void push(uint64_t t, uint32_t p) noexcept {
        data_.push_back({t, p, 0});
        is_sorted_ = false;
    }
    inline const HeapNode& top() noexcept {
        if (!is_sorted_) {
            std::sort(data_.begin(), data_.end(),
                      [](const HeapNode& a, const HeapNode& b) { return a.arrival_time > b.arrival_time; });
            is_sorted_ = true;
        }
        return data_.back();
    }
    inline void pop() noexcept { data_.pop_back(); }
    inline bool empty() const noexcept { return data_.empty(); }
};

// ============================================================================
// 4. BENCHMARK SCENARIOS
// ============================================================================

template <class Scheduler>
void BM_Burst(benchmark::State& state) {
    const size_t batch_size = state.range(0);
    std::mt19937_64 gen(42);
    std::vector<uint64_t> times(batch_size);
    for (auto& t : times)
        t = gen();

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
// 5. REGISTRATION MATRIX
// ============================================================================

#define REGISTER_RANGE(CLASS, NAME)                                                               \
    BENCHMARK(BM_Churn<CLASS>)->Name("Churn_" NAME)->RangeMultiplier(8)->Range(128, 1024 * 1024); \
    BENCHMARK(BM_Burst<CLASS>)->Name("Burst_" NAME)->RangeMultiplier(8)->Range(128, 1024 * 1024);

// Baselines
REGISTER_RANGE(StdBinaryHeap, "Std_Binary");
REGISTER_RANGE(BoostPairingHeap, "Boost_Pairing");
REGISTER_RANGE(VectorSortScheduler, "Challenger_VectorSort");

// Boost D-Ary
REGISTER_RANGE(BoostDAryHeap<4>, "Boost_D4");
REGISTER_RANGE(BoostDAryHeap<8>, "Boost_D8");
REGISTER_RANGE(BoostDAryHeap<16>, "Boost_D16");

// Titan Hand-Tuned Templates
REGISTER_RANGE(FastDAryHeap<2>, "Titan_D2");
REGISTER_RANGE(FastDAryHeap<4>, "Titan_D4");
REGISTER_RANGE(FastDAryHeap<8>, "Titan_D8");
REGISTER_RANGE(FastDAryHeap<16>, "Titan_D16");
REGISTER_RANGE(FastDAryHeap<32>, "Titan_D32");
REGISTER_RANGE(FastDAryHeap<64>, "Titan_D64");

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    std::cout << "-----------------------------------------------------------\n";
    std::cout << "TITAN SCHEDULER CROSS-ARITY HARDWARE SEARCH\n";
    std::cout << "Testing optimal D-factor for L1/L2 cache prefetching.\n";
    std::cout << "-----------------------------------------------------------\n";
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}