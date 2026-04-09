#include "titan/core/lob_state.hpp"

namespace titan::core {

// ============================================================================
// CORE LOB OPERATIONS
// ============================================================================

template <uint32_t RingSize>
void LOBState<RingSize>::add_order(OrderId id, Price price, Quantity qty, uint8_t side, OrderPoolAllocator& pool) {
    const Handle h = static_cast<Handle>(id & 0xFFFFFFFF);
    OrderNode& node = pool.get_node(h);

    // Initialize the payload. Next pointer must be strictly nullified as it's a tail insertion.
    node.price = price;
    node.quantity = qty;
    node.side = side;
    node.next = NULL_HANDLE;

    if (is_hot(price)) {
        // --- HOT ZONE: O(1) Cache-resident insertion ---
        PriceLevel& level = get_level_for_write(price);

        if (level.tail == NULL_HANDLE) {
            // Scenario A: First order at this price level.
            level.head = h;
            level.tail = h;
            node.prev = NULL_HANDLE;

            // Hardware Acceleration: Ignite the bitmasks for O(1) scan capabilities.
            if (side == 0) {  // Assuming 0 = BID
                set_active_bid(price);
            } else {  // Assuming 1 = ASK
                set_active_ask(price);
            }
        } else {
            // Scenario B: Queue exists (FIFO). Append to the tail of the intrusive list.
            node.prev = level.tail;
            pool.get_node(level.tail).next = h;
            level.tail = h;
        }

        level.total_qty += qty;

    } else {
        // --- COLD ZONE: O(log N) Red-Black Tree fallback ---
        auto& target_map = (side == 0) ? cold_bids_ : cold_asks_;

        // Map lookup (will default-construct a PriceLevel if the price doesn't exist).
        PriceLevel& level = target_map[price];

        // Intrusive linked-list logic mirrors the Hot Zone.
        if (level.tail == NULL_HANDLE) {
            level.head = h;
            level.tail = h;
            node.prev = NULL_HANDLE;
        } else {
            node.prev = level.tail;
            pool.get_node(level.tail).next = h;
            level.tail = h;
        }

        level.total_qty += qty;
    }
}

template <uint32_t RingSize>
void LOBState<RingSize>::cancel_order(OrderId id, OrderPoolAllocator& pool) {
    // 1. Unpack the 64-bit ID into its physical handle and generation tag.
    const Handle h = static_cast<Handle>(id & 0xFFFFFFFF);
    const uint32_t expected_gen = static_cast<uint32_t>(id >> 32);

    OrderNode& node = pool.get_node(h);

    // 2. ABA Problem / Stale Execution Protection.
    // If the generation doesn't match, this order was already completely filled
    // or cancelled, and the memory handle has been reused by a new order.
    if (node.generation != expected_gen) {
        return;
    }

    const Price price = node.price;
    const uint8_t side = node.side;
    const Quantity qty = node.quantity;

    // 3. Dispatch based on memory boundaries.
    if (is_hot(price)) {
        // --- HOT ZONE: O(1) Cache-resident removal ---

        // Direct array access. We avoid get_level_for_write() here because
        // if this is a phantom price, we don't want to accidentally clear it.
        PriceLevel& level = hot_levels_[get_index(price)];

        // Safety check against lazy clearing overlaps
        if (level.actual_price != price)
            return;

        // O(1) Intrusive doubly-linked list unlinking
        if (node.prev != NULL_HANDLE) {
            pool.get_node(node.prev).next = node.next;
        } else {
            level.head = node.next;  // Node was the head
        }

        if (node.next != NULL_HANDLE) {
            pool.get_node(node.next).prev = node.prev;
        } else {
            level.tail = node.prev;  // Node was the tail
        }

        // Subtract liquidity
        level.total_qty -= qty;

        // Hardware Acceleration: Extinguish the bitmask if the price level is now dead.
        // This ensures get_best_bid() / get_best_ask() won't stop at an empty level.
        if (level.total_qty == 0) {
            if (side == 0)
                set_empty_bid(price);  // Assuming 0 = BID
            else
                set_empty_ask(price);  // Assuming 1 = ASK
        }

    } else {
        // --- COLD ZONE: O(log N) Red-Black Tree removal ---
        auto& target_map = (side == 0) ? cold_bids_ : cold_asks_;
        auto it = target_map.find(price);

        if (it != target_map.end()) {
            PriceLevel& level = it->second;

            // List unlinking logic is identical to the Hot Zone
            if (node.prev != NULL_HANDLE) {
                pool.get_node(node.prev).next = node.next;
            } else {
                level.head = node.next;
            }

            if (node.next != NULL_HANDLE) {
                pool.get_node(node.next).prev = node.prev;
            } else {
                level.tail = node.prev;
            }

            level.total_qty -= qty;

            // Memory cleanup: Completely erase the map node if empty
            // to maintain accurate reverse iteration in get_best_bid/ask.
            if (level.total_qty == 0) {
                target_map.erase(it);
            }
        }
    }

    // Important Architectural Note:
    // LOBState DOES NOT call `pool.free(h)`.
    // Freeing memory is the strict responsibility of the MatchingEngine,
    // which allows the engine to emit cancel execution reports before the memory dies.
}

// ============================================================================
// HARDWARE ACCELERATED SEARCH (O(1))
// ============================================================================

template <uint32_t RingSize>
Price LOBState<RingSize>::get_best_ask() const noexcept {
    // ASK -> Lowest possible price. We scan FORWARD looking for the first set bit.

    const uint32_t anchor_idx = get_index(anchor_price_);
    const uint32_t start_l1 = anchor_idx >> 6;
    const uint32_t start_l2 = start_l1 >> 6;

    // Helper lambda to scan a specific range of the L2/L1 bitmasks
    auto scan_forward = [&](uint32_t l2_start, uint32_t l2_end, uint32_t base_offset) -> Price {
        for (uint32_t i = l2_start; i <= l2_end; ++i) {
            uint64_t mask = l2_mask_asks_[i];

            // Mask out stale bits physically located before the logical start
            if (i == start_l2 && base_offset == 0) {
                mask &= ~((1ULL << (start_l1 & 63)) - 1);
            }

            if (mask == 0)
                continue;  // Skip 4096 empty price levels in 1 CPU cycle

            // Found an active L1 chunk
            const uint32_t l1_bit = std::countr_zero(mask);
            const uint32_t l1_idx = (i << 6) + l1_bit;

            uint64_t final_mask = l1_mask_asks_[l1_idx];

            // Mask out stale bits physically located before the logical start in L1
            if (l1_idx == start_l1 && base_offset == 0) {
                final_mask &= ~((1ULL << (anchor_idx & 63)) - 1);
            }

            if (final_mask != 0) {
                // Found the exact active price level
                const uint32_t final_bit = std::countr_zero(final_mask);
                const uint32_t buffer_idx = (l1_idx << 6) + final_bit;
                return anchor_price_ + buffer_idx - anchor_idx + base_offset;
            }
        }
        return UINT32_MAX;
    };

    // Phase 1: Scan from anchor to the physical end of the Ring Buffer
    Price best = scan_forward(start_l2, L2_SIZE - 1, 0);
    if (best != UINT32_MAX)
        return best;

    // Phase 2: Wrap around and scan from physical index 0 up to the anchor
    if (start_l2 > 0) {
        const uint32_t offset = RingSize - anchor_idx;
        best = scan_forward(0, start_l2 - 1, offset);
        if (best != UINT32_MAX)
            return best;
    }

    // Phase 3: Hot Zone is entirely empty. Query the Cold Zone Red-Black Tree.
    // std::map is sorted ascending, so begin() is the lowest price.
    return cold_asks_.empty() ? UINT32_MAX : cold_asks_.begin()->first;
}

template <uint32_t RingSize>
Price LOBState<RingSize>::get_best_bid() const noexcept {
    // BID -> Highest possible price. We scan BACKWARDS looking for the highest set bit.

    const uint32_t anchor_idx = get_index(anchor_price_);

    // The maximum possible logical price currently inside the Hot Zone
    const uint32_t max_hot_price = anchor_price_ + RingSize - 1;
    const uint32_t max_idx = get_index(max_hot_price);

    const uint32_t start_l1 = anchor_idx >> 6;
    const uint32_t start_l2 = start_l1 >> 6;

    const uint32_t end_l1 = max_idx >> 6;
    const uint32_t end_l2 = end_l1 >> 6;

    // Helper lambda to scan backwards using std::countl_zero
    auto scan_backward = [&](int32_t l2_start, int32_t l2_end, uint32_t max_allowed_idx) -> Price {
        for (int32_t i = l2_start; i >= l2_end; --i) {
            uint64_t mask = l2_mask_bids_[i];

            // Mask out bits that logically exceed our Hot Zone ceiling
            if (i == static_cast<int32_t>(max_allowed_idx >> 12)) {
                const uint32_t limit_bit = (max_allowed_idx >> 6) & 63;
                if (limit_bit < 63) {
                    mask &= (1ULL << (limit_bit + 1)) - 1;
                }
            }

            if (mask == 0)
                continue;

            // Find highest active bit in L2
            const uint32_t l1_bit = 63 - std::countl_zero(mask);
            const uint32_t l1_idx = (i << 6) + l1_bit;

            uint64_t final_mask = l1_mask_bids_[l1_idx];

            // Mask out bits in L1 that exceed the ceiling
            if (l1_idx == (max_allowed_idx >> 6)) {
                const uint32_t limit_bit = max_allowed_idx & 63;
                if (limit_bit < 63) {
                    final_mask &= (1ULL << (limit_bit + 1)) - 1;
                }
            }

            if (final_mask != 0) {
                // Find highest active bit in L1
                const uint32_t final_bit = 63 - std::countl_zero(final_mask);
                const uint32_t buffer_idx = (l1_idx << 6) + final_bit;

                // Un-wrap the buffer index back into an absolute price
                if (buffer_idx >= anchor_idx) {
                    return anchor_price_ + (buffer_idx - anchor_idx);
                } else {
                    return anchor_price_ + (RingSize - anchor_idx + buffer_idx);
                }
            }
        }
        return 0;  // Not found in this segment
    };

    Price best = 0;

    // Determine if the logical window wraps around the physical array end
    if (end_l2 < start_l2) {
        // Phase 1: Scan from physical max_idx down to 0
        best = scan_backward(end_l2, 0, max_idx);
        if (best != 0)
            return best;

        // Phase 2: Wrap around and scan from the physical end down to anchor
        best = scan_backward(L2_SIZE - 1, start_l2, RingSize - 1);
        if (best != 0)
            return best;
    } else {
        // Window does not wrap around physically
        best = scan_backward(end_l2, start_l2, max_idx);
        if (best != 0)
            return best;
    }

    // Phase 3: Hot Zone empty. Query Cold Zone.
    // std::map is sorted ascending, so rbegin() gives the highest price.
    return cold_bids_.empty() ? 0 : cold_bids_.rbegin()->first;
}

// ============================================================================
// MARKET SHIFTING (Overflow Resolution)
// ============================================================================

template <uint32_t RingSize>
void LOBState<RingSize>::shift_window(Price new_anchor) noexcept {
    if (new_anchor == anchor_price_)
        return;

    const Price old_anchor = anchor_price_;
    anchor_price_ = new_anchor;

    // --- CRITICAL FIX: Calculate the eviction boundaries ---
    Price drop_start = 0;
    Price drop_end = 0;

    if (new_anchor > old_anchor) {
        // Market moved UP: Prices at the bottom of the old window are evicted
        drop_start = old_anchor;
        drop_end = std::min(new_anchor - 1, old_anchor + RingSize - 1);
    } else {
        // Market moved DOWN: Prices at the top of the old window are evicted
        drop_start = new_anchor + RingSize;
        drop_end = old_anchor + RingSize - 1;
    }

    // 1. Identify and Evict prices that are no longer in the Hot Zone
    for (Price p = drop_start; p <= drop_end; ++p) {
        const uint32_t idx = get_index(p);
        PriceLevel& level = hot_levels_[idx];

        // Lazy Clear verification: Only evict if the level actually belongs to this price
        if (level.actual_price == p && level.total_qty > 0) {
            const uint32_t l1_idx = idx >> 6;
            const uint64_t bit_mask = 1ULL << (idx & 63);

            // ZERO-MEMORY-ACCESS CHECK: Query hardware masks to determine order side
            if (l1_mask_bids_[l1_idx] & bit_mask) {
                cold_bids_[p] = level;
                set_empty_bid(p);
            } else if (l1_mask_asks_[l1_idx] & bit_mask) {
                cold_asks_[p] = level;
                set_empty_ask(p);
            }

            // Purge the level data
            level.total_qty = 0;
            level.head = NULL_HANDLE;
            level.tail = NULL_HANDLE;
        }
    }

    // 2. Absorb prices from Cold Zone that now fit into the new Hot Zone window
    // We use lower_bound to efficiently find prices within the [anchor, anchor + RingSize) range

    // Process Bids from Cold Zone
    auto bit = cold_bids_.lower_bound(new_anchor);
    while (bit != cold_bids_.end() && bit->first < new_anchor + RingSize) {
        Price p = bit->first;
        PriceLevel& level = get_level_for_write(p);
        level = bit->second;

        set_active_bid(p);
        bit = cold_bids_.erase(bit);  // Move from map to ring buffer
    }

    // Process Asks from Cold Zone
    auto ait = cold_asks_.lower_bound(new_anchor);
    while (ait != cold_asks_.end() && ait->first < new_anchor + RingSize) {
        Price p = ait->first;
        PriceLevel& level = get_level_for_write(p);
        level = ait->second;

        set_active_ask(p);
        ait = cold_asks_.erase(ait);
    }
}

// ============================================================================
// EXPLICIT TEMPLATE INSTANTIATION
// ============================================================================

template class LOBState<detail::FINAL_RING_SIZE>;

}  // namespace titan::core