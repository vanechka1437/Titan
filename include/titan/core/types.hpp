#pragma once
#include <cstdint>

namespace titan::core {

using Handle = int32_t;
constexpr Handle NULL_HANDLE = -1;

using Price = uint32_t;
using OrderQty = int64_t;
using OrderId = uint64_t;
using OwnerId = uint16_t;

struct alignas(32) OrderNode {
    OrderId id;
    OrderQty quantity;
    Handle next;
    Handle prev;
    Price price;
    OwnerId owner_id;
    uint8_t side;

    uint8_t _padding[1]{0};
};

}  // namespace titan::core