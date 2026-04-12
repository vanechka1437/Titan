#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "titan/core/lob_state.hpp"
#include "titan/core/memory.hpp"
#include "titan/core/types.hpp"

namespace titan::core {

// ============================================================================
// 1. MARKET DATA EVENT
// Extremely lightweight POD structure to communicate LOB changes back to the
// agents' ShadowLOBs and account balances without dynamic allocations.
// ============================================================================
struct alignas(32) MarketDataEvent {
    enum class Type : uint8_t { TRADE = 0, LOB_UPDATE = 1 };

    Type type;
    uint8_t side;       // 0: Bid, 1: Ask
    uint16_t owner_id;  // Agent who owns the triggered order
    Price price;
    int32_t qty_delta;   // Positive for additions, negative for cancels/trades
    int64_t cash_delta;  // For TRADE events: how much money was exchanged
};

// ============================================================================
// EVENT BUFFER (Templated for user control)
// Defaulting to 8192 (262 KB), enough to sweep thousands of LOB levels.
// ============================================================================
template <uint32_t MaxEvents = 8192>
struct EventBuffer {
    static constexpr uint32_t MAX_EVENTS = MaxEvents;

    std::array<MarketDataEvent, MAX_EVENTS> events;
    uint32_t count{0};

    inline void clear() noexcept { count = 0; }

    inline void push_update(uint8_t side, Price price, int32_t qty_delta) {
        if (count >= MAX_EVENTS) [[unlikely]] {
            throw std::runtime_error("EventBuffer overflow: Market sweep exceeded MAX_EVENTS capacity.");
        }
        events[count++] = {MarketDataEvent::Type::LOB_UPDATE, side, 0, price, qty_delta, 0};
    }

    inline void push_trade(uint8_t side, uint16_t owner_id, Price price, OrderQty qty) {
        if (count >= MAX_EVENTS) [[unlikely]] {
            throw std::runtime_error("EventBuffer overflow: Market sweep exceeded MAX_EVENTS capacity.");
        }
        events[count++] = {MarketDataEvent::Type::TRADE,
                           side,
                           owner_id,
                           price,
                           -(static_cast<int32_t>(qty)),
                           (side == 0) ? -(static_cast<int64_t>(price) * qty) : (static_cast<int64_t>(price) * qty)};
    }
};

// Чтобы не делать весь MatchingEngine шаблонным классом, создадим удобный алиас
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
    void execute_trade(Handle maker_handle, uint16_t taker_id, uint8_t side, OrderQty remaining_qty,
                       DefaultEventBuffer& out_events);

public:
    explicit MatchingEngine(OrderPoolAllocator& pool, uint32_t max_orders)
        : pool_(pool), order_map_(max_orders, NULL_HANDLE) {}

    // 1. Process new Limit/Market Orders.
    // This function will check for spread crossing, call execute_trade() in a loop
    // if necessary, and push the remainder to lob_.add_order().
    void process_order(uint64_t order_id, uint16_t owner_id, uint8_t side, Price price, OrderQty qty,
                       DefaultEventBuffer& out_events);

    // 2. Process Cancellations for O(1) lookup and removal
    void process_cancel(uint64_t target_order_id, DefaultEventBuffer& out_events);

    // 3. Fast reset between RL episodes
    void reset() noexcept;

    // Const accessor for the BatchSimulator to peek at best prices if needed
    [[nodiscard]] inline const OptimalLOBState& get_lob() const noexcept { return lob_; }
};

}  // namespace titan::core