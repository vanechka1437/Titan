#pragma once
#include <cstdint>

namespace titan::core {

using Handle = int32_t;
constexpr Handle NULL_HANDLE = -1;

using Price = uint32_t;
using OrderQty = int32_t;
using OrderId = uint64_t;

struct alignas(32) OrderNode {
    Handle next;
    Handle prev;
    uint32_t generation;
    uint32_t id;
    int32_t owner_id;
    Price price;
    OrderQty quantity;
    uint8_t side;

    uint8_t _padding1[3];
};

}  // namespace titan::core