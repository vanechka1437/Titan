#pragma once
#include <cstdint>

namespace titan::core {

using Handle = int32_t;
constexpr Handle NULL_HANDLE = -1;

struct alignas(32) OrderNode {
    Handle next;
    Handle prev;
    uint32_t generation;
    int32_t owner_id;
    float price;
    int32_t volume;

    uint32_t _padding1;
    uint32_t _padding2;
};
}