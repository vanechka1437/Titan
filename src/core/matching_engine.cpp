#include "titan/core/matching_engine.hpp"

#include <algorithm>
#include "titan/core/state.hpp"

namespace titan::core {

// ============================================================================
// Internal Execution Helper
// ============================================================================
template <uint32_t ObsDepth>
void MatchingEngine::execute_trade(Handle maker_handle, uint16_t taker_id, uint8_t taker_side, OrderQty& remaining_qty,
                                   EnvironmentState<ObsDepth>& state) {
    OrderNode& maker = pool_.get_node(maker_handle);
    
    // Determine how much can be traded
    OrderQty trade_qty = std::min(remaining_qty, maker.quantity);

    lob_.reduce_level_qty(maker.side, maker.price, trade_qty);

    // 1. Decrease the volume of the limit order (Maker)
    maker.quantity -= trade_qty;

    // 2. Generate a TRADE event for the maker (passive side)
    state.record_trade(maker.price, trade_qty, maker.owner_id, taker_id, maker.side);

    // 3. Update the maker's ActiveOrderRecord tracking
    state.update_active_order_qty(maker.owner_id, maker.id, maker.quantity);

    // 4. Update the remaining aggressive volume
    remaining_qty -= trade_qty;
}

// ============================================================================
// Process Limit / Market Orders
// ============================================================================
template <uint32_t ObsDepth>
void MatchingEngine::process_order(uint16_t owner_id, uint8_t side, Price price, OrderQty qty,
                                   EnvironmentState<ObsDepth>& state) {

    OrderQty remaining_qty = qty;

    if (side == 0) {  // --- INCOMING BID (Aggressive Buy) ---
        while (remaining_qty > 0) {
            Price best_ask = lob_.get_best_ask();

            // Spread check: If best ask is strictly greater than our buy limit, matching stops.
            if (best_ask > price) {
                break;
            }

            // Retrieve the head of the FIFO queue at the best ask price
            Handle maker_handle = lob_.get_first_order(1, best_ask);
            if (maker_handle == NULL_HANDLE) {
                break;  // Failsafe against phantom bits or state desync
            }

            OrderNode& maker = pool_.get_node(maker_handle);

            // SELF-TRADE PREVENTION (STP) - Cancel Resting Policy
            if (maker.owner_id == owner_id) [[unlikely]] {
                // Generate CANCEL event
                state.record_cancel(maker.price, maker.quantity, maker.owner_id, maker.side, maker.id);
                // Free the tracking slot
                state.clear_active_order(maker.owner_id, maker.id);
                
                lob_.remove_order(maker_handle, pool_);
                pool_.free(maker_handle);
                continue;  // Skip execution and proceed to the next resting order
            }

            // Execute the cross trade (modifies remaining_qty by reference)
            execute_trade(maker_handle, owner_id, side, remaining_qty, state);

            // If the passive maker order is completely depleted, remove it permanently
            if (maker.quantity == 0) {
                state.clear_active_order(maker.owner_id, maker.id);
                lob_.remove_order(maker_handle, pool_);
                pool_.free(maker_handle);
            }
        }
    } else {  // --- INCOMING ASK (Aggressive Sell) ---
        while (remaining_qty > 0) {
            Price best_bid = lob_.get_best_bid();

            // Spread check: If best bid is strictly less than our sell limit, matching stops.
            if (best_bid < price) {
                break;
            }

            Handle maker_handle = lob_.get_first_order(0, best_bid);
            if (maker_handle == NULL_HANDLE) {
                break;
            }

            OrderNode& maker = pool_.get_node(maker_handle);

            // SELF-TRADE PREVENTION (STP) - Cancel Resting Policy
            if (maker.owner_id == owner_id) [[unlikely]] {
                state.record_cancel(maker.price, maker.quantity, maker.owner_id, maker.side, maker.id);
                state.clear_active_order(maker.owner_id, maker.id);
                
                lob_.remove_order(maker_handle, pool_);
                pool_.free(maker_handle);
                continue;
            }

            execute_trade(maker_handle, owner_id, side, remaining_qty, state);

            if (maker.quantity == 0) {
                state.clear_active_order(maker.owner_id, maker.id);
                lob_.remove_order(maker_handle, pool_);
                pool_.free(maker_handle);
            }
        }
    }

    // 3. PASSIVE ORDER ROUTING (Resting Limit Order)
    if (remaining_qty > 0 && price != 0 && price != UINT32_MAX) {
        
        // Let LOBState forge the Smart ID
        OrderId smart_id = lob_.add_order(owner_id, price, remaining_qty, side, pool_);

        if (smart_id != 0) [[likely]] {
            // Track the order in the Zero-Copy Array so Python knows its ID
            state.add_active_order(owner_id, smart_id, remaining_qty);

            // Broadcast the acceptance
            state.record_accepted(price, remaining_qty, owner_id, side, smart_id);
        }
    }
}

// ============================================================================
// Process Cancellations (O(1) Smart ID Lookup + ABA Protection)
// ============================================================================
template <uint32_t ObsDepth>
void MatchingEngine::process_cancel(OrderId target_order_id, uint16_t requesting_owner_id,
                                    EnvironmentState<ObsDepth>& state) {
    if (target_order_id == 0) [[unlikely]] {
        return; // Reject empty ID
    }

    // 1. Unpack the Smart ID
    const Handle h = extract_handle(target_order_id);
    const Generation target_gen = extract_generation(target_order_id);

    // 2. Bounds checking (Pool capacity)
    if (h >= pool_.capacity()) [[unlikely]] {
        return;
    }

    OrderNode& node = pool_.get_node(h);

    // 3. SECURE ABA PROTECTION 
    // If the generation counters do not match, or the physical memory ID differs,
    // this memory cell has already been reused by another order. The cancel is stale.
    if (node.generation != target_gen || node.id != target_order_id) [[unlikely]] {
        return;
    }

    // 4. Ownership Verification
    if (node.owner_id != requesting_owner_id) [[unlikely]] {
        return;  // Reject malicious/buggy cancellation attempts
    }

    const uint8_t side = node.side;
    const Price p = node.price;
    const OrderQty q = node.quantity;

    // 5. Unlink the order from the LOB doubly-linked list
    lob_.remove_order(h, pool_);

    // 6. Free the memory back to the LIFO pool (increments Generation)
    pool_.free(h);

    // 7. Clear Zero-Copy tracking array
    state.clear_active_order(requesting_owner_id, target_order_id);

    // 8. Generate an event for Python to observe
    state.record_cancel(p, q, requesting_owner_id, side, target_order_id);
}

// ============================================================================
// Reset
// ============================================================================
void MatchingEngine::reset() noexcept {
    lob_.reset();
}

// ============================================================================
// EXPLICIT TEMPLATE INSTANTIATION FOR METHODS
// ============================================================================
// We explicitly instantiate the methods for the default observation depth of 20
template void MatchingEngine::execute_trade<DEFAULT_OBS_DEPTH>(Handle, uint16_t, uint8_t, OrderQty&, EnvironmentState<DEFAULT_OBS_DEPTH>&);
template void MatchingEngine::process_order<DEFAULT_OBS_DEPTH>(uint16_t, uint8_t, Price, OrderQty, EnvironmentState<DEFAULT_OBS_DEPTH>&);
template void MatchingEngine::process_cancel<DEFAULT_OBS_DEPTH>(OrderId, uint16_t, EnvironmentState<DEFAULT_OBS_DEPTH>&);

}  // namespace titan::core