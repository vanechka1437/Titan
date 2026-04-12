#include "titan/core/matching_engine.hpp"

#include <algorithm>

namespace titan::core {

// ============================================================================
// Internal Execution Helper
// ============================================================================
void MatchingEngine::execute_trade(Handle maker_handle, uint16_t taker_id, uint8_t taker_side, OrderQty trade_qty,
                                   EventBuffer& out_events) noexcept {
    OrderNode& maker = pool_.get_node(maker_handle);

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
                                   EventBuffer& out_events) noexcept {
    OrderQty remaining_qty = qty;

    if (side == 0) {  // --- INCOMING BID (Aggressive Buy) ---
        while (remaining_qty > 0) {
            Price best_ask = lob_.get_best_ask();

            // If the spread doesn't cross (best ask is higher than our price), exit the matching loop
            if (best_ask > price) {
                break;
            }

            // Get the first order (head of the queue) at the best ask price
            Handle maker_handle = lob_.get_first_order(1, best_ask);
            if (maker_handle == NULL_HANDLE) {
                break;  // Protection against phantom bits/desync
            }

            OrderNode& maker = pool_.get_node(maker_handle);
            OrderQty trade_qty = std::min(remaining_qty, maker.quantity);

            // Execute the trade
            execute_trade(maker_handle, owner_id, side, trade_qty, out_events);
            remaining_qty -= trade_qty;

            // If the passive order is fully filled, completely remove it from the LOB
            if (maker.quantity == 0) {
                order_map_[maker.id] = NULL_HANDLE;      // 1. Remove from the O(1) mapping
                lob_.remove_order(maker_handle, pool_);  // 2. Unlink from the intrusive list
                pool_.free(maker_handle);                // 3. IMPORTANT: Return the handle back to the free pool
            }
        }
    } else {  // --- INCOMING ASK (Aggressive Sell) ---
        while (remaining_qty > 0) {
            Price best_bid = lob_.get_best_bid();

            // If the spread doesn't cross (best bid is lower than our price), exit
            if (best_bid < price) {
                break;
            }

            Handle maker_handle = lob_.get_first_order(0, best_bid);
            if (maker_handle == NULL_HANDLE) {
                break;
            }

            OrderNode& maker = pool_.get_node(maker_handle);
            OrderQty trade_qty = std::min(remaining_qty, maker.quantity);

            execute_trade(maker_handle, owner_id, side, trade_qty, out_events);
            remaining_qty -= trade_qty;

            if (maker.quantity == 0) {
                order_map_[maker.id] = NULL_HANDLE;
                lob_.remove_order(maker_handle, pool_);
                pool_.free(maker_handle);  // RETURN MEMORY TO POOL
            }
        }
    }

    // If remaining volume exists after aggressive matching (or no spread crossed at all),
    // place the remainder into the order book as a new passive Limit Order.
    if (remaining_qty > 0) {
        Handle new_handle = lob_.add_order(order_id, owner_id, price, remaining_qty, side, pool_);

        if (new_handle != NULL_HANDLE) [[likely]] {
            order_map_[order_id] = new_handle;  // Store the Handle for O(1) cancellation

            // Broadcast to the world that new liquidity appeared in the LOB
            out_events.push_update(side, price, remaining_qty);
        }
    }
}

// ============================================================================
// Process Cancellations (O(1) Memory Lookup)
// ============================================================================
void MatchingEngine::process_cancel(uint64_t target_order_id, EventBuffer& out_events) noexcept {
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