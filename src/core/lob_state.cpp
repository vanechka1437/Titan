#include "titan/core/lob_state.hpp"

namespace titan::core {

// ============================================================================
// CORE LOB OPERATIONS
// ============================================================================

template <uint32_t RingSize>
Handle LOBState<RingSize>::add_order(OrderId id, uint16_t owner_id, Price price, OrderQty qty, uint8_t side,
                                     OrderPoolAllocator& pool) {
    // 1. Secure O(1) allocation from the LIFO free list (preserves ABA protection tags)
    const Handle h = pool.allocate();
    if (h == NULL_HANDLE) [[unlikely]] {
        return NULL_HANDLE;  // Pool capacity exhausted
    }

    OrderNode& node = pool.get_node(h);

    // 2. Initialize the payload. Next pointer must be strictly nullified as it's a tail insertion.
    node.id = id;
    node.owner_id = owner_id;
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

    // Return the memory handle back to the MatchingEngine for O(1) mapping
    return h;
}

template <uint32_t RingSize>
inline void LOBState<RingSize>::reduce_level_qty(uint8_t side, Price price, OrderQty trade_qty) noexcept {
    if (is_hot(price)) {
        // --- HOT ZONE ---
        PriceLevel& level = hot_levels_[get_index(price)];

        if (level.actual_price != price) {
            return;
        }

        level.total_qty -= trade_qty;

        if (level.total_qty <= 0) {
            level.total_qty = 0;
            if (side == 0) {
                set_empty_bid(price);
            } else {
                set_empty_ask(price);
            }
        }
    } else {
        // --- COLD ZONE ---
        auto& target_map = (side == 0) ? cold_bids_ : cold_asks_;
        auto it = target_map.find(price);

        if (it != target_map.end()) {
            it->second.total_qty -= trade_qty;

            if (it->second.total_qty <= 0) {
                it->second.total_qty = 0;
            }
        }
    }
}

template <uint32_t RingSize>
void LOBState<RingSize>::remove_order(Handle h, OrderPoolAllocator& pool) noexcept {
    if (h == NULL_HANDLE) [[unlikely]] {
        return;
    }

    OrderNode& node = pool.get_node(h);

    const Price price = node.price;
    const uint8_t side = node.side;
    const OrderQty qty = node.quantity;

    // Dispatch based on memory boundaries.
    if (is_hot(price)) {
        // --- HOT ZONE: O(1) Cache-resident removal ---

        // Direct array access. We avoid get_level_for_write() here because
        // if this is a phantom price, we don't want to accidentally clear it.
        PriceLevel& level = hot_levels_[get_index(price)];

        // Safety check against lazy clearing overlaps
        if (level.actual_price != price) {
            return;
        }

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
            if (side == 0) {
                set_empty_bid(price);  // Assuming 0 = BID
            } else {
                set_empty_ask(price);  // Assuming 1 = ASK
            }
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
    // ASK -> Lowest possible price. Scan FORWARD.
    const uint32_t anchor_idx = get_index(anchor_price_);
    const uint64_t max_hot_price_64 = static_cast<uint64_t>(anchor_price_) + RingSize - 1;
    const uint32_t max_idx = get_index(static_cast<uint32_t>(max_hot_price_64));

    auto scan_forward = [&](uint32_t from_idx, uint32_t to_idx) -> Price {
        if (from_idx > to_idx)
            return UINT32_MAX;

        const uint32_t start_l2 = from_idx >> 12;
        const uint32_t end_l2 = to_idx >> 12;

        for (uint32_t i = start_l2; i <= end_l2; ++i) {
            uint64_t mask = l2_mask_asks_[i];

            // Mask out bits strictly before from_idx
            if (i == (from_idx >> 12)) {
                const uint32_t lower_bit = (from_idx >> 6) & 63;
                if (lower_bit > 0)
                    mask &= ~((1ULL << lower_bit) - 1);
            }
            // Mask out bits strictly after to_idx
            if (i == (to_idx >> 12)) {
                const uint32_t limit_bit = (to_idx >> 6) & 63;
                if (limit_bit < 63)
                    mask &= (1ULL << (limit_bit + 1)) - 1;
            }

            // Keep extracting bits until the L2 mask is exhausted
            while (mask != 0) {
                const uint32_t l1_bit = std::countr_zero(mask);
                const uint32_t l1_idx = (i << 6) + l1_bit;
                uint64_t final_mask = l1_mask_asks_[l1_idx];

                // Precise bounds checking inside L1
                if (l1_idx == (from_idx >> 6)) {
                    const uint32_t lower_bit = from_idx & 63;
                    if (lower_bit > 0)
                        final_mask &= ~((1ULL << lower_bit) - 1);
                }
                if (l1_idx == (to_idx >> 6)) {
                    const uint32_t limit_bit = to_idx & 63;
                    if (limit_bit < 63)
                        final_mask &= (1ULL << (limit_bit + 1)) - 1;
                }

                if (final_mask != 0) {
                    const uint32_t final_bit = std::countr_zero(final_mask);
                    const uint32_t buffer_idx = (l1_idx << 6) + final_bit;

                    if (buffer_idx >= anchor_idx) {
                        return anchor_price_ + (buffer_idx - anchor_idx);
                    } else {
                        return anchor_price_ + (RingSize - anchor_idx + buffer_idx);
                    }
                }

                // If L1 was empty due to masking, clear the bit and retry L2
                mask &= ~(1ULL << l1_bit);
            }
        }
        return UINT32_MAX;
    };

    Price best = UINT32_MAX;

    if (max_idx < anchor_idx) {
        // Physical wrap-around
        best = scan_forward(anchor_idx, RingSize - 1);
        if (best != UINT32_MAX)
            return best;

        best = scan_forward(0, max_idx);
        if (best != UINT32_MAX)
            return best;
    } else {
        // Continuous block
        best = scan_forward(anchor_idx, max_idx);
        if (best != UINT32_MAX)
            return best;
    }

    return cold_asks_.empty() ? UINT32_MAX : cold_asks_.begin()->first;
}

template <uint32_t RingSize>
Price LOBState<RingSize>::get_best_bid() const noexcept {
    // BID -> Highest possible price. Scan BACKWARDS.
    const uint32_t anchor_idx = get_index(anchor_price_);
    const uint64_t max_hot_price_64 = static_cast<uint64_t>(anchor_price_) + RingSize - 1;
    const uint32_t max_idx = get_index(static_cast<uint32_t>(max_hot_price_64));

    auto scan_backward = [&](uint32_t from_idx, uint32_t to_idx) -> Price {
        if (from_idx < to_idx)
            return 0;

        const int32_t start_l2 = from_idx >> 12;
        const int32_t end_l2 = to_idx >> 12;

        for (int32_t i = start_l2; i >= end_l2; --i) {
            uint64_t mask = l2_mask_bids_[i];

            if (i == static_cast<int32_t>(from_idx >> 12)) {
                const uint32_t limit_bit = (from_idx >> 6) & 63;
                if (limit_bit < 63)
                    mask &= (1ULL << (limit_bit + 1)) - 1;
            }
            if (i == static_cast<int32_t>(to_idx >> 12)) {
                const uint32_t lower_bit = (to_idx >> 6) & 63;
                if (lower_bit > 0)
                    mask &= ~((1ULL << lower_bit) - 1);
            }

            while (mask != 0) {
                const uint32_t l1_bit = 63 - std::countl_zero(mask);
                const uint32_t l1_idx = (i << 6) + l1_bit;
                uint64_t final_mask = l1_mask_bids_[l1_idx];

                if (l1_idx == (from_idx >> 6)) {
                    const uint32_t limit_bit = from_idx & 63;
                    if (limit_bit < 63)
                        final_mask &= (1ULL << (limit_bit + 1)) - 1;
                }
                if (l1_idx == (to_idx >> 6)) {
                    const uint32_t lower_bit = to_idx & 63;
                    if (lower_bit > 0)
                        final_mask &= ~((1ULL << lower_bit) - 1);
                }

                if (final_mask != 0) {
                    const uint32_t final_bit = 63 - std::countl_zero(final_mask);
                    const uint32_t buffer_idx = (l1_idx << 6) + final_bit;

                    if (buffer_idx >= anchor_idx) {
                        return anchor_price_ + (buffer_idx - anchor_idx);
                    } else {
                        return anchor_price_ + (RingSize - anchor_idx + buffer_idx);
                    }
                }

                mask &= ~(1ULL << l1_bit);
            }
        }
        return 0;
    };

    Price best = 0;

    if (max_idx < anchor_idx) {
        // Physical wrap-around (logical highs are physically at 0..max_idx)
        best = scan_backward(max_idx, 0);
        if (best != 0)
            return best;

        best = scan_backward(RingSize - 1, anchor_idx);
        if (best != 0)
            return best;
    } else {
        // Continuous block
        best = scan_backward(max_idx, anchor_idx);
        if (best != 0)
            return best;
    }

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
        if (p == drop_end)
            break;
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

template <uint32_t RingSize>
void LOBState<RingSize>::shift_window_to_center(Price target_price) noexcept {
    // Calculate new anchor so that target_price is perfectly centered within the Ring Buffer
    Price new_anchor = (target_price > RingSize / 2) ? (target_price - RingSize / 2) : 0;

    if (new_anchor == anchor_price_)
        return;

    const Price old_anchor = anchor_price_;
    const Price old_end = old_anchor + RingSize - 1;
    const Price new_end = new_anchor + RingSize - 1;

    // Helper to evict levels that fall out of the newly calculated intersection
    auto evict_range = [&](Price start, Price end) {
        for (Price p = start;; ++p) {
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

            // Break condition placed at the end to process the boundary inclusively
            // while completely preventing uint32_t overflow on ++p
            if (p == end)
                break;
        }
    };

    // 1. Evict out-of-bounds data based on the direction of the market shift
    if (new_anchor > old_anchor) {
        evict_range(old_anchor, std::min(old_end, new_anchor - 1));
    } else {
        evict_range(std::max(old_anchor, new_end + 1), old_end);
    }

    // Commit the new anchor offset
    anchor_price_ = new_anchor;

    // Helper to safely absorb valid levels from the Cold Zone (Boost Flat Map)
    auto absorb_range = [&](Price start, Price end, auto& cold_map, bool is_bid) {
        auto first = cold_map.lower_bound(start);
        auto last = first;

        // Phase 1: Transfer memory representations to the Hot Zone and ignite bitmasks
        while (last != cold_map.end() && last->first <= end) {
            Price p = last->first;
            PriceLevel& level = get_level_for_write(p);
            level = last->second;

            if (is_bid)
                set_active_bid(p);
            else
                set_active_ask(p);

            ++last;  // Advance iterator without erasing to prevent quadratic shifting
        }

        // Phase 2: CRITICAL O(N) ERASE
        // boost::container::flat_map uses a contiguous array. Erasing elements one-by-one
        // inside the loop causes catastrophic O(N^2) memory shifting overhead.
        // Range erase guarantees a single, highly-optimized block shift.
        if (first != last) {
            cold_map.erase(first, last);
        }
    };

    // 2. Absorb newly covered data from the fallback trees
    if (new_anchor > old_anchor) {
        absorb_range(std::max(new_anchor, old_end + 1), new_end, cold_bids_, true);
        absorb_range(std::max(new_anchor, old_end + 1), new_end, cold_asks_, false);
    } else {
        absorb_range(new_anchor, std::min(new_end, old_anchor - 1), cold_bids_, true);
        absorb_range(new_anchor, std::min(new_end, old_anchor - 1), cold_asks_, false);
    }
}

template <uint32_t RingSize>
Handle LOBState<RingSize>::get_first_order(uint8_t side, Price price) const noexcept {
    if (is_hot(price)) {
        const PriceLevel& level = hot_levels_[get_index(price)];
        if (level.actual_price == price) {
            return level.head;
        }
    } else {
        const auto& target_map = (side == 0) ? cold_bids_ : cold_asks_;
        auto it = target_map.find(price);
        if (it != target_map.end()) {
            return it->second.head;
        }
    }
    return NULL_HANDLE;
}

template <uint32_t RingSize>
void LOBState<RingSize>::reset() noexcept {
    // 1. Zero out hardware bitmasks (instantly clears the Hot Zone logic)
    std::fill(std::begin(l1_mask_bids_), std::end(l1_mask_bids_), 0);
    std::fill(std::begin(l2_mask_bids_), std::end(l2_mask_bids_), 0);
    std::fill(std::begin(l1_mask_asks_), std::end(l1_mask_asks_), 0);
    std::fill(std::begin(l2_mask_asks_), std::end(l2_mask_asks_), 0);

    // 2. We don't need to loop over hot_levels_ because actual_price lazy clearing
    // will naturally overwrite old data when new prices arrive.
    // However, if we want strict determinism across RL episodes, we can force reset the anchor.
    anchor_price_ = 0;

    // 3. Clear the Cold Zone maps
    cold_bids_.clear();
    cold_asks_.clear();
}
// ============================================================================
// EXPLICIT TEMPLATE INSTANTIATION
// ============================================================================

template class LOBState<detail::FINAL_RING_SIZE>;

}  // namespace titan::core