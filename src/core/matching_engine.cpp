#include "titan/core/matching_engine.hpp"

#include <algorithm>

namespace titan::core {

// ============================================================================
// Internal Execution Helper
// ============================================================================
void MatchingEngine::execute_trade(Handle maker_handle, OwnerId taker_id, uint8_t taker_side, OrderQty& remaining_qty,
                                   EventList& out_events) {
    OrderNode& maker = pool_.get_node(maker_handle);
    
    // Determine how much can be traded
    OrderQty trade_qty = std::min(remaining_qty, maker.quantity);

    lob_.reduce_level_qty(maker.side, maker.price, trade_qty);

    // 1. Decrease the volume of the limit order (Maker)
    maker.quantity -= trade_qty;

    // 2. Generate a TRADE event for the maker (passive side)
    out_events.push_back({
        maker.id,                      // order_id
        -trade_qty,                    // qty_delta
        maker.price,                   // price
        maker.owner_id,                // owner_id (maker)
        taker_id,                      // taker_id
        MarketDataEvent::Type::TRADE,  // type
        maker.side                     // side
    });

    // 3. Update the remaining aggressive volume
    remaining_qty -= trade_qty;
}

// ============================================================================
// Process Limit / Market Orders
// ============================================================================
void MatchingEngine::process_order(OwnerId owner_id, uint8_t side, Price price, OrderQty qty,
                                   EventList& out_events) {

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
                out_events.push_back({
                    maker.id,
                    static_cast<OrderQty>(-maker.quantity), // Kept from your implementation
                    maker.price,
                    maker.owner_id,
                    0,
                    MarketDataEvent::Type::CANCEL,
                    maker.side
                });
                
                lob_.remove_order(maker_handle, pool_);
                pool_.free(maker_handle);
                continue;  // Skip execution and proceed to the next resting order
            }

            // Execute the cross trade (modifies remaining_qty by reference)
            execute_trade(maker_handle, owner_id, side, remaining_qty, out_events);

            // If the passive maker order is completely depleted, remove it permanently
            if (maker.quantity == 0) {
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
                out_events.push_back({
                    maker.id,
                    static_cast<OrderQty>(-maker.quantity),
                    maker.price,
                    maker.owner_id,
                    0,
                    MarketDataEvent::Type::CANCEL,
                    maker.side
                });
                
                lob_.remove_order(maker_handle, pool_);
                pool_.free(maker_handle);
                continue;
            }

            execute_trade(maker_handle, owner_id, side, remaining_qty, out_events);

            if (maker.quantity == 0) {
                lob_.remove_order(maker_handle, pool_);
                pool_.free(maker_handle);
            }
        }
    }

    // --- 3. PASSIVE ORDER ROUTING & REJECT HANDLING ---
    // If we still have quantity left after attempting to cross the spread
    if (remaining_qty > 0) {
        
        // Limit Order check: Price must be valid (Not 0 and not UINT32_MAX)
        if (price != 0 && price != UINT32_MAX) {
            
            // Let LOBState forge the Smart ID
            OrderId smart_id = lob_.add_order(owner_id, price, remaining_qty, side, pool_);

            if (smart_id != 0) [[likely]] {
                // Broadcast the acceptance (Successfully added to order book)
                out_events.push_back({
                    smart_id,
                    remaining_qty,
                    price,
                    owner_id,
                    0,
                    MarketDataEvent::Type::ACCEPTED,
                    side
                });
            } else {
                // Memory Pool Exhaustion! 
                // Cannot add to book. Reject the order so the agent wakes up.
                out_events.push_back({
                    0, // No valid ID
                    remaining_qty,
                    price,
                    owner_id,
                    0,
                    MarketDataEvent::Type::REJECTED,
                    side
                });
            }
        } else {
            // Ghost Order Prevention!
            // It's a Market Order (or IOC) that couldn't be fully filled because 
            // the opposite book is empty. Reject the unfilled remainder.
            out_events.push_back({
                0, // Market orders don't have a resting ID
                remaining_qty,
                price,
                owner_id,
                0,
                MarketDataEvent::Type::REJECTED,
                side
            });
        }
    }
}

// ============================================================================
// Process Cancellations (O(1) Smart ID Lookup + ABA Protection)
// ============================================================================
void MatchingEngine::process_cancel(OrderId target_order_id, OwnerId requesting_owner_id,
                                    EventList& out_events) {
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

    // 7. Generate an event for Python to observe
    out_events.push_back({
        target_order_id,
        -q,
        p,
        requesting_owner_id,
        0,
        MarketDataEvent::Type::CANCEL,
        side
    });
}

// ============================================================================
// Reset
// ============================================================================
void MatchingEngine::reset() noexcept {
    lob_.reset();
}

}  // namespace titan::core