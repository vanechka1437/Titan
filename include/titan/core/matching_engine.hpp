#pragma once

#include <cstdint>
#include <vector>

#include "titan/core/lob_state.hpp"
#include "titan/core/types.hpp"

namespace titan::core {

// Forward declaration to prevent circular dependency with memory.hpp
class OrderPoolAllocator;

// ============================================================================
// MATCHING ENGINE
// The algorithmic controller. It owns the OptimalLOBState, crosses the spread,
// generates execution events, and delegates pure storage logic to LOBState.
// Pure physical DES component: completely decoupled from PyTorch Zero-Copy logic.
// ============================================================================
class MatchingEngine {
private:
    OrderPoolAllocator& pool_;
    OptimalLOBState lob_;

    // --- Internal Execution Helpers ---

    // Core function to execute a trade between an incoming order and an existing resting order
    void execute_trade(Handle maker_handle, OwnerId taker_id, uint8_t taker_side, OrderQty& remaining_qty,
                       EventList& out_events);

public:
    // Relies strictly on O(1) Smart IDs provided by the memory pool
    explicit MatchingEngine(OrderPoolAllocator& pool)
        : pool_(pool) {}

    // Disabled copying
    MatchingEngine(const MatchingEngine&) = delete;
    MatchingEngine& operator=(const MatchingEngine&) = delete;

    // 1. Process new Limit/Market Orders.
    // This function will check for spread crossing, call execute_trade() in a loop
    // if necessary, and push the remainder to lob_.add_order().
    void process_order(OwnerId owner_id, uint8_t side, Price price, OrderQty qty,
                       EventList& out_events);

    // 2. Process Cancellations for O(1) lookup and removal with ABA protection
    void process_cancel(OrderId target_order_id, OwnerId requesting_owner_id, 
                        EventList& out_events);

    // 3. Fast reset between RL episodes
    void reset() noexcept;

    // Const accessor for reading raw liquidity
    [[nodiscard]] inline const OptimalLOBState& get_lob() const noexcept { return lob_; }
};

}  // namespace titan::core