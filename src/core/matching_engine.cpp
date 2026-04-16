#include "titan/core/matching_engine.hpp"

#include <algorithm>

namespace titan::core {

// ============================================================================
// Internal Execution Helper
// ============================================================================
void MatchingEngine::execute_trade(Handle maker_handle, uint16_t taker_id, uint8_t taker_side, OrderQty trade_qty,
                                   DefaultEventBuffer& out_events) {
    OrderNode& maker = pool_.get_node(maker_handle);

    lob_.reduce_level_qty(maker.side, maker.price, trade_qty);

    // 1. Decrease the volume of the limit order (Maker)
    maker.quantity -= trade_qty;

    // 2. Generate a TRADE event for the maker (passive side)
    out_events.push_trade(maker.side, maker.owner_id, maker.price, trade_qty);

    // 3. Generate a TRADE event for the taker (aggressive side)
    out_events.push_trade(taker_side, taker_id, maker.price, trade_qty);

    // 4. Generate a LOB_UPDATE so all agents' shadow order books reflect
    // the decreased liquidity at this specific price level
    out_events.push_update(maker.side, maker.price, -(static_cast<int32_t>(trade_qty)));
}

// ============================================================================
// Process Limit / Market Orders
// ============================================================================
void MatchingEngine::process_order(uint64_t order_id, uint16_t owner_id, uint8_t side, Price price, OrderQty qty,
                                   DefaultEventBuffer& out_events) {
    // 1. SEGFAULT PROTECTION (ДОЛЖНА БЫТЬ ПЕРВОЙ!)
    // Ensure the Python-generated order_id does not exceed our allocated vector capacity.
    if (order_id >= order_map_.size()) [[unlikely]] {
        return;
    }

    // 2. DUPLICATE ORDER PROTECTION
    // Reject duplicate order IDs that are already active in the book.
    if (order_map_[order_id] != NULL_HANDLE) [[unlikely]] {
        return;
    }

    OrderQty remaining_qty = qty;

    if (side == 0) {  // --- INCOMING BID (Aggressive Buy) ---
        while (remaining_qty > 0) {
            Price best_ask = lob_.get_best_ask();

            // Spread check: If best ask is strictly greater than our buy limit, matching stops.
            // For Market Orders (price = UINT32_MAX), this condition is theoretically never met
            // until the book is completely empty (best_ask = UINT32_MAX).
            if (best_ask > price) {
                break;
            }

            // Retrieve the head of the FIFO queue at the best ask price
            Handle maker_handle = lob_.get_first_order(1, best_ask);
            if (maker_handle == NULL_HANDLE) {
                break;  // Failsafe against phantom bits or state desync
            }

            OrderNode& maker = pool_.get_node(maker_handle);

            // 2. SELF-TRADE PREVENTION (STP) - Cancel Resting Policy
            // If the aggressive order matches against a passive order from the same agent,
            // we cancel the passive maker order to prevent artificial volume generation (wash trading).
            if (maker.owner_id == owner_id) [[unlikely]] {
                out_events.push_update(maker.side, maker.price, -(static_cast<int32_t>(maker.quantity)));
                order_map_[maker.id] = NULL_HANDLE;
                lob_.remove_order(maker_handle, pool_);
                pool_.free(maker_handle);
                continue;  // Skip execution and proceed to the next resting order
            }

            OrderQty trade_qty = std::min(remaining_qty, maker.quantity);

            // Execute the cross trade
            execute_trade(maker_handle, owner_id, side, trade_qty, out_events);
            remaining_qty -= trade_qty;

            // If the passive maker order is completely depleted, remove it permanently
            if (maker.quantity == 0) {
                order_map_[maker.id] = NULL_HANDLE;
                lob_.remove_order(maker_handle, pool_);
                pool_.free(maker_handle);
            }
        }
    } else {  // --- INCOMING ASK (Aggressive Sell) ---
        while (remaining_qty > 0) {
            Price best_bid = lob_.get_best_bid();

            // Spread check: If best bid is strictly less than our sell limit, matching stops.
            // For Market Orders (price = 0), this condition is never met until the book is empty.
            if (best_bid < price) {
                break;
            }

            Handle maker_handle = lob_.get_first_order(0, best_bid);
            if (maker_handle == NULL_HANDLE) {
                break;
            }

            OrderNode& maker = pool_.get_node(maker_handle);

            // 2. SELF-TRADE PREVENTION (STP) - Cancel Resting Policy
            if (maker.owner_id == owner_id) [[unlikely]] {
                out_events.push_update(maker.side, maker.price, -(static_cast<int32_t>(maker.quantity)));
                order_map_[maker.id] = NULL_HANDLE;
                lob_.remove_order(maker_handle, pool_);
                pool_.free(maker_handle);
                continue;
            }

            OrderQty trade_qty = std::min(remaining_qty, maker.quantity);

            execute_trade(maker_handle, owner_id, side, trade_qty, out_events);
            remaining_qty -= trade_qty;

            if (maker.quantity == 0) {
                order_map_[maker.id] = NULL_HANDLE;
                lob_.remove_order(maker_handle, pool_);
                pool_.free(maker_handle);
            }
        }
    }

    // 3. PASSIVE ORDER ROUTING & MARKET ORDER SENTINEL PROTECTION
    // If volume remains (e.g., partial fill or no crossing spread), we insert it as a resting Limit Order.
    // Sentinel values (price == 0 for Market Sells, price == UINT32_MAX for Market Buys) are strictly
    // prohibited from entering the book. Their unfilled remainder simply evaporates (FOK/IOC behavior).
    if (remaining_qty > 0 && price != 0 && price != UINT32_MAX) {
        Handle new_handle = lob_.add_order(order_id, owner_id, price, remaining_qty, side, pool_);

        if (new_handle != NULL_HANDLE) [[likely]] {
            order_map_[order_id] = new_handle;  // Register handle for O(1) cancellation routing

            // Broadcast the liquidity addition to the external world
            out_events.push_update(side, price, remaining_qty);
        }
    }
}

// ============================================================================
// Process Cancellations (O(1) Memory Lookup)
// ============================================================================
void MatchingEngine::process_cancel(uint64_t target_order_id, uint16_t requesting_owner_id,
                                    DefaultEventBuffer& out_events) {
    // Bounds checking protection (in case of an invalid ID from the agent)
    if (target_order_id >= order_map_.size()) [[unlikely]] {
        return;
    }

    Handle h = order_map_[target_order_id];

    // If handle is NULL, the order was already fully filled or previously cancelled
    if (h == NULL_HANDLE) {
        return;
    }

    OrderNode& node = pool_.get_node(h);
    if (node.owner_id != requesting_owner_id) [[unlikely]] {
        return;  // Reject cancellation attempts from agents who do not own the order
    }
    uint8_t side = node.side;
    Price p = node.price;
    OrderQty q = node.quantity;

    // 1. Unlink the order from the LOB doubly-linked list
    lob_.remove_order(h, pool_);

    // 2. Clear the O(1) mapping
    order_map_[target_order_id] = NULL_HANDLE;

    // 3. IMPORTANT: Free the memory back to the LIFO pool
    pool_.free(h);

    // 4. Generate an event for Shadow LOBs (so agents see the liquidity removal)
    out_events.push_update(side, p, -(static_cast<int32_t>(q)));
}

// ============================================================================
// Reset (End of RL Episode)
// ============================================================================
void MatchingEngine::reset() noexcept {
    // Reset the internal state of the LOB structure
    lob_.reset();

    // Nullify all order tracking pointers
    std::fill(order_map_.begin(), order_map_.end(), NULL_HANDLE);

    // Note: The memory pool (OrderPoolAllocator) is NOT reset here,
    // as it is managed and reset globally by the Master Owner (UnifiedMemoryArena::reset())
}

}  // namespace titan::core