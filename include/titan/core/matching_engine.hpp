#pragma once

#include <cstdint>
#include <vector>

#include "titan/core/lob_state.hpp"
#include "titan/core/memory.hpp"
#include "titan/core/types.hpp"

namespace titan::core {

// ============================================================================
// 1. MARKET DATA EVENT
// Extremely lightweight POD structure to communicate LOB changes back to the
// agents' ShadowLOBs and account balances. Aligned for optimal cache utilization.
// ============================================================================
struct alignas(32) MarketDataEvent {
    enum class Type : uint8_t { TRADE = 0, LOB_UPDATE = 1 };

    OrderQty qty_delta;   // Positive for additions, negative for cancels/trades
    OrderQty cash_delta;  // For TRADE events: how much money was exchanged
    Price price;
    OwnerId owner_id;  // Agent who owns the triggered order
    Type type;
    uint8_t side;  // 0: Bid, 1: Ask

    uint8_t _padding[8]{0};  // Padding to make the struct 32 bytes
};

// ============================================================================
// EVENT BUFFER
// Replaced fixed std::array with std::vector to provide dynamic reallocation.
// This completely eliminates std::runtime_error exceptions during extreme
// AI-generated market sweeps (e.g., a 10M lot fat finger order), protecting
// multi-day RL training sessions from crashing.
// ============================================================================
template <uint32_t InitialCapacity = 8192>
class EventBuffer {
private:
    std::vector<MarketDataEvent> events_;

public:
    EventBuffer() {
        // Pre-allocate to prevent dynamic allocation on the hot path for 99.9% of steps.
        events_.reserve(InitialCapacity);
    }

    inline void clear() noexcept { events_.clear(); }

    inline size_t size() const noexcept { return events_.size(); }

    inline const MarketDataEvent& operator[](size_t index) const noexcept { return events_[index]; }

    // Pushes safely. If capacity is exceeded, std::vector reallocates automatically
    // without crashing the environment.
    inline void push_update(uint8_t side, Price price, OrderQty qty_delta) {
        events_.push_back({qty_delta, 0, price, 0, MarketDataEvent::Type::LOB_UPDATE, side});
    }

    inline void push_trade(uint8_t side, OwnerId owner_id, Price price, OrderQty qty) {
        events_.push_back({(side == 0) ? static_cast<OrderQty>(qty) : -(static_cast<OrderQty>(qty)),
                           (side == 0) ? -(static_cast<OrderQty>(price) * qty) : (static_cast<OrderQty>(price) * qty),
                           price, owner_id, MarketDataEvent::Type::TRADE, side});
    }
};

// Alias to prevent matching engine from becoming a template
using DefaultEventBuffer = EventBuffer<8192>;

// ============================================================================
// 2. MATCHING ENGINE
// The algorithmic controller. It owns the OptimalLOBState, crosses the spread,
// generates execution events, and delegates pure storage logic to LOBState.
// ============================================================================
class MatchingEngine {
private:
    OrderPoolAllocator& pool_;
    OptimalLOBState lob_;

    // Array mapping OrderID directly to memory Handle for O(1) cancels.
    // If we have 1,000,000 max orders, this vector takes ~4MB per environment.
    std::vector<Handle> order_map_;

    // --- Internal Execution Helpers ---

    // Core function to execute a trade between an incoming order and an existing resting order
    void execute_trade(Handle maker_handle, OwnerId taker_id, uint8_t side, OrderQty remaining_qty,
                       DefaultEventBuffer& out_events);

public:
    explicit MatchingEngine(OrderPoolAllocator& pool, uint32_t max_orders)
        : pool_(pool), order_map_(max_orders, NULL_HANDLE) {}

    // 1. Process new Limit/Market Orders.
    // This function will check for spread crossing, call execute_trade() in a loop
    // if necessary, and push the remainder to lob_.add_order().
    void process_order(OrderId order_id, OwnerId owner_id, uint8_t side, Price price, OrderQty qty,
                       DefaultEventBuffer& out_events);

    // 2. Process Cancellations for O(1) lookup and removal
    void process_cancel(OrderId target_order_id, OwnerId requesting_owner_id, DefaultEventBuffer& out_events);

    // 3. Fast reset between RL episodes
    void reset() noexcept;

    // Const accessor for the BatchSimulator to peek at best prices if needed
    [[nodiscard]] inline const OptimalLOBState& get_lob() const noexcept { return lob_; }
};

}  // namespace titan::core