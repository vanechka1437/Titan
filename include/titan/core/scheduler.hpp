#pragma once

#include <algorithm>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <vector>

#include "titan/core/types.hpp"

namespace titan::core {

// ============================================================================
// 1. FAT EVENT PAYLOAD (For Routing logic, not for sorting)
// Stored in a separate pool to keep the sorting heap ultra-light (16 bytes).
// Compressed to 48 bytes using an anonymous union to fit within a single Cache Line.
// ============================================================================
struct alignas(8) ScheduledEvent {
    enum class Type : uint8_t { AGENT_WAKEUP, ORDER_ARRIVAL, MARKET_DATA };

    uint64_t time;        // 8 bytes
    uint32_t target_id;   // 4 bytes
    Type type;            // 1 byte
    uint8_t _padding[3];  // 3 bytes padding for 8-byte alignment

    // Anonymous union: ActionPayload and MarketDataEvent share the same 32 bytes of memory.
    // Safe because both are strictly POD types.
    union {
        ActionPayload action;
        MarketDataEvent market_data;
    };

    // Factories
    static inline ScheduledEvent agent_wakeup(uint64_t t, uint32_t agent) noexcept {
        ScheduledEvent e{}; 
        e.time = t; 
        e.target_id = agent; 
        e.type = Type::AGENT_WAKEUP; 
        return e;
    }
    static inline ScheduledEvent order_arrival(uint64_t t, uint32_t agent, const ActionPayload& act) noexcept {
        ScheduledEvent e{}; 
        e.time = t; 
        e.target_id = agent; 
        e.type = Type::ORDER_ARRIVAL; 
        e.action = act; 
        return e;
    }
    static inline ScheduledEvent market_data(uint64_t t, uint32_t agent, const MarketDataEvent& md) noexcept {
        ScheduledEvent e{}; 
        e.time = t; 
        e.target_id = agent; 
        e.type = Type::MARKET_DATA; 
        e.market_data = md; 
        return e;
    }
};
static_assert(sizeof(ScheduledEvent) == 48, "ScheduledEvent must be 48 bytes to fit in a cache line");

// ============================================================================
// 2. THE ROUTING KEY (Metadata)
// Size: Exactly 16 bytes.
// Four nodes fit into a single 64-byte cache line for massive SIMD throughput.
// ============================================================================
struct alignas(16) HeapNode {
    uint64_t arrival_time;  // Primary Min-Heap sorting key
    uint32_t payload_idx;   // Index into the ScheduledEvent Pool
    uint32_t sequence_id;   // Strict FIFO tie-breaker for simultaneous events
};
static_assert(sizeof(HeapNode) == 16, "HeapNode must be exactly 16 bytes");

// ============================================================================
// 3. FAST D-ARY HEAP (Templated)
// O(logD N) Priority Queue with Stable FIFO Sorting.
// Recommended Arity: 4 or 8 for modern CPUs.
// ============================================================================
template <uint32_t Arity>
class FastDAryHeap {
    static_assert(Arity >= 2, "Arity must be at least 2");
    static_assert((Arity & (Arity - 1)) == 0, "Arity must be a power of 2 for optimal indexing");

private:
    // Aligned to 64 bytes to prevent "Cache Line Splits" during child scans
    alignas(64) HeapNode* data_{nullptr};
    std::size_t size_{0};
    std::size_t max_capacity_;

    // Monotonically increasing counter to guarantee FIFO determinism
    uint32_t sequence_counter_{0};

    // --- Strict Deterministic Comparisons ---
    static inline bool is_better_or_equal(const HeapNode& a, const HeapNode& b) noexcept {
        if (a.arrival_time != b.arrival_time)
            return a.arrival_time < b.arrival_time;
        return a.sequence_id <= b.sequence_id;
    }

    static inline bool is_strictly_better(const HeapNode& a, const HeapNode& b) noexcept {
        if (a.arrival_time != b.arrival_time)
            return a.arrival_time < b.arrival_time;
        return a.sequence_id < b.sequence_id;
    }

    /**
     * @brief Tournament-style branchless min-search.
     * Compiler unrolls this entirely based on the Arity template parameter.
     */
    inline std::size_t find_best_child(std::size_t first_child) const noexcept {
        if constexpr (Arity == 8) {
            // Level 1: 4 pairs
            std::size_t a =
                is_better_or_equal(data_[first_child + 0], data_[first_child + 1]) ? first_child + 0 : first_child + 1;
            std::size_t b =
                is_better_or_equal(data_[first_child + 2], data_[first_child + 3]) ? first_child + 2 : first_child + 3;
            std::size_t c =
                is_better_or_equal(data_[first_child + 4], data_[first_child + 5]) ? first_child + 4 : first_child + 5;
            std::size_t d =
                is_better_or_equal(data_[first_child + 6], data_[first_child + 7]) ? first_child + 6 : first_child + 7;
            // Level 2: 2 pairs
            std::size_t e = is_better_or_equal(data_[a], data_[b]) ? a : b;
            std::size_t f = is_better_or_equal(data_[c], data_[d]) ? c : d;
            // Level 3: Final winner
            return is_better_or_equal(data_[e], data_[f]) ? e : f;
        } else if constexpr (Arity == 4) {
            std::size_t a =
                is_better_or_equal(data_[first_child + 0], data_[first_child + 1]) ? first_child + 0 : first_child + 1;
            std::size_t b =
                is_better_or_equal(data_[first_child + 2], data_[first_child + 3]) ? first_child + 2 : first_child + 3;
            return is_better_or_equal(data_[a], data_[b]) ? a : b;
        } else {
            // Generic fallback for other arities
            std::size_t best = first_child;
            for (std::size_t i = 1; i < Arity; ++i) {
                if (is_strictly_better(data_[first_child + i], data_[best])) {
                    best = first_child + i;
                }
            }
            return best;
        }
    }

    inline void sift_up(std::size_t idx) noexcept {
        HeapNode node = data_[idx];
        while (idx > 0) {
            std::size_t parent = (idx - 1) / Arity;
            if (is_better_or_equal(data_[parent], node))
                break;
            data_[idx] = data_[parent];
            idx = parent;
        }
        data_[idx] = node;
    }

    inline void sift_down(std::size_t idx) noexcept {
        HeapNode node = data_[idx];

        while (true) {
            const std::size_t first_child = Arity * idx + 1;
            if (first_child >= size_)
                break;

            // Tournament-style branchless selection
            std::size_t best = find_best_child(first_child);

            if (is_better_or_equal(node, data_[best]))
                break;

            data_[idx] = data_[best];
            idx = best;
        }
        data_[idx] = node;
    }

public:
    explicit FastDAryHeap(std::size_t max_capacity) : max_capacity_(max_capacity) {
        // Allocate space + Arity-sized buffer for Sentinels (prevents bounds checks)
        const std::size_t alloc_size = max_capacity_ + Arity;
        
        // Use standard C++17 aligned allocation for cross-platform compatibility (x86/ARM/Apple Silicon)
        data_ = static_cast<HeapNode*>(::operator new[](alloc_size * sizeof(HeapNode), std::align_val_t{64}));

        if (!data_)
            throw std::bad_alloc();
            
        clear();
    }

    ~FastDAryHeap() {
        if (data_) {
            ::operator delete[](data_, std::align_val_t{64});
        }
    }

    // Standard HFT practice: Non-copyable to ensure strict ownership of the arena
    FastDAryHeap(const FastDAryHeap&) = delete;
    FastDAryHeap& operator=(const FastDAryHeap&) = delete;

    [[nodiscard]] inline bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] inline std::size_t size() const noexcept { return size_; }
    [[nodiscard]] inline const HeapNode& top() const noexcept { return data_[0]; }

    // Returns false on overflow (Silent drop for RL sandbox safety) instead of crashing
    inline bool push(uint64_t time, uint32_t payload_idx) noexcept {
        if (size_ >= max_capacity_) [[unlikely]] {
            return false;
        }

        const std::size_t idx = size_++;
        // Guarantee strict FIFO via sequence_counter_
        data_[idx] = {time, payload_idx, sequence_counter_++};

        // Maintain Sentinel Zone: ensures sift_down never compares against garbage memory.
        // The next Arity elements are always set to infinity (and max sequence).
        for (uint32_t i = 0; i < Arity; ++i) {
            data_[size_ + i] = {0xFFFFFFFFFFFFFFFFULL, 0, 0xFFFFFFFF};
        }

        sift_up(idx);
        return true;
    }

    inline void pop() noexcept {
        if (size_ == 0) [[unlikely]]
            return;

        data_[0] = data_[--size_];

        // Ensure the vacated spot becomes a perfect sentinel
        data_[size_] = {0xFFFFFFFFFFFFFFFFULL, 0, 0xFFFFFFFF};

        if (size_ > 0) {
            sift_down(0);
        }
    }

    inline void clear() noexcept {
        size_ = 0;
        sequence_counter_ = 0;  // Reset monotonic ID for the next RL episode

        // Initialize the first two cache lines of sentinels
        for (uint32_t i = 0; i < Arity * 2; ++i) {
            data_[i] = {0xFFFFFFFFFFFFFFFFULL, 0, 0xFFFFFFFF};
        }
    }
};

// ============================================================================
// 4. FAST SCHEDULER (Combines the 16-byte Heap with a Payload Pool)
// ============================================================================
class FastScheduler {
private:
    FastDAryHeap<4> heap_;
    std::vector<ScheduledEvent> payloads_;
    std::vector<uint32_t> free_list_;

public:
    explicit FastScheduler(uint32_t max_capacity = 65536) : heap_(max_capacity) {
        payloads_.reserve(max_capacity);
        free_list_.reserve(max_capacity);
    }

    inline void push(const ScheduledEvent& ev) noexcept {
        uint32_t idx;
        if (!free_list_.empty()) {
            idx = free_list_.back();
            free_list_.pop_back();
            payloads_[idx] = ev;
        } else {
            idx = static_cast<uint32_t>(payloads_.size());
            payloads_.push_back(ev);
        }
        heap_.push(ev.time, idx);
    }

    [[nodiscard]] inline const ScheduledEvent& top() const noexcept {
        return payloads_[heap_.top().payload_idx];
    }

    inline void pop() noexcept {
        free_list_.push_back(heap_.top().payload_idx);
        heap_.pop();
    }

    [[nodiscard]] inline bool empty() const noexcept { 
        return heap_.empty(); 
    }

    inline void clear() noexcept {
        heap_.clear();
        payloads_.clear();
        free_list_.clear();
    }
};

}  // namespace titan::core