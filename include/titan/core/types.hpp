#pragma once

#include <cstdint>
#include <limits>

namespace titan::core {

inline constexpr uint32_t DEFAULT_OBS_DEPTH = 20;

using Handle = uint32_t;
constexpr Handle NULL_HANDLE = std::numeric_limits<Handle>::max();

using Price = uint32_t;
using OrderQty = int64_t;
using OrderId = uint64_t;
using OwnerId = uint16_t;
using Generation = uint32_t;

// ============================================================================
// SMART ORDER ID HELPERS (Zero-Memory-Lookup Routing)
// OrderId (64-bit) = [ Generation (32-bit) ] | [ Handle (32-bit) ]
// ============================================================================
[[nodiscard]] inline OrderId pack_order_id(Generation gen, Handle handle) noexcept {
    return (static_cast<OrderId>(gen) << 32) | static_cast<OrderId>(handle);
}

[[nodiscard]] inline Handle extract_handle(OrderId id) noexcept {
    return static_cast<Handle>(id & 0xFFFFFFFF);
}

[[nodiscard]] inline Generation extract_generation(OrderId id) noexcept {
    return static_cast<Generation>(id >> 32);
}

// ============================================================================
// 1. LIMIT ORDER BOOK NODE (Intrusive Linked List Element)
// ============================================================================
struct alignas(32) OrderNode {
    // Hardware/Routing info using anonymous union for zero-overhead casting
    // Note: Assumes Little-Endian architecture (standard for x86_64/ARM64)
    union {
        OrderId id;               // 8 bytes (Smart ID: Gen + Handle)
        struct {
            Handle handle_id;     // 4 bytes (Lower 32 bits)
            Generation generation;// 4 bytes (Upper 32 bits, ABA Protection Counter)
        };
    };
    
    // Payload
    OrderQty quantity;            // 8 bytes
    
    // Intrusive LOB pointers
    Handle next;                  // 4 bytes
    Handle prev;                  // 4 bytes
    Price price;                  // 4 bytes
    OwnerId owner_id;             // 2 bytes
    uint8_t side;                 // 1 byte (0: Bid, 1: Ask)
    uint8_t _padding[1]{0};       // 1 byte padding to hit exactly 32 bytes
};

// ============================================================================
// 2. ACTIVE ORDER TRACKER (Zero-Copy Export for Python)
// Allows Python agents to know their resting orders without HashMaps
// ============================================================================
struct alignas(16) ActiveOrderRecord {
    OrderId id;           // 8 bytes (0 means this slot is empty)
    OrderQty quantity;    // 8 bytes (Remaining quantity in the book)
};

}  // namespace titan::core