#pragma once

#include <cstdint>

#include "titan/core/lob_state.hpp"
#include "titan/core/memory.hpp"
#include "titan/core/types.hpp"

namespace titan::core {

// Forward declaration of EnvironmentState for Zero-Copy event routing
template <uint32_t ObsDepth>
class EnvironmentState;

// ============================================================================
// 1. MARKET DATA EVENT
// Extremely lightweight POD structure to communicate LOB changes back to the
// agents' ShadowLOBs and account balances. Aligned for optimal cache utilization.
// ============================================================================
struct alignas(32) MarketDataEvent {
    enum class Type : uint8_t { TRADE = 0, LOB_UPDATE = 1, CANCEL = 2, ACCEPTED = 3 };

    OrderQty qty_delta;   // 8 bytes: Positive for additions, negative for cancels/trades
    OrderQty cash_delta;  // 8 bytes: For TRADE events: how much money was exchanged
    Price price;          // 4 bytes
    OwnerId owner_id;     // 2 bytes: Agent who owns the triggered order
    Type type;            // 1 byte
    uint8_t side;         // 1 byte: 0: Bid, 1: Ask
    
    // Replaced 8-byte padding with OrderId to support Zero-Copy Active Order tracking
    OrderId order_id;     // 8 bytes: Smart ID (Generation + Handle)
};
static_assert(sizeof(MarketDataEvent) == 32, "MarketDataEvent must be exactly 32 bytes for cache alignment");

// ============================================================================
// 2. MATCHING ENGINE
// The algorithmic controller. It owns the OptimalLOBState, crosses the spread,
// generates execution events, and delegates pure storage logic to LOBState.
// ============================================================================
class MatchingEngine {
private:
    OrderPoolAllocator& pool_;
    OptimalLOBState lob_;

    // --- Internal Execution Helpers ---

    // Core function to execute a trade between an incoming order and an existing resting order
    template <uint32_t ObsDepth>
    void execute_trade(Handle maker_handle, OwnerId taker_id, uint8_t side, OrderQty& remaining_qty,
                       EnvironmentState<ObsDepth>& state);

public:
    // No longer allocates order_map_, relying strictly on O(1) Smart IDs
    explicit MatchingEngine(OrderPoolAllocator& pool)
        : pool_(pool) {}

    // Disabled copying
    MatchingEngine(const MatchingEngine&) = delete;
    MatchingEngine& operator=(const MatchingEngine&) = delete;

    // 1. Process new Limit/Market Orders.
    // This function will check for spread crossing, call execute_trade() in a loop
    // if necessary, and push the remainder to lob_.add_order().
    template <uint32_t ObsDepth>
    void process_order(OwnerId owner_id, uint8_t side, Price price, OrderQty qty,
                       EnvironmentState<ObsDepth>& state);

    // 2. Process Cancellations for O(1) lookup and removal
    template <uint32_t ObsDepth>
    void process_cancel(OrderId target_order_id, OwnerId requesting_owner_id, 
                        EnvironmentState<ObsDepth>& state);

    // 3. Fast reset between RL episodes
    void reset() noexcept;

    // Const accessor for the BatchSimulator and State to read LOB data
    [[nodiscard]] inline const OptimalLOBState& get_lob() const noexcept { return lob_; }
};

}  // namespace titan::core