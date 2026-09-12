/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_SHM_NUM_UTIL_H
#define SHMEM_SHM_NUM_UTIL_H

#include <type_traits>
#include <limits>
#include <string>
#include <cctype>
#include <cstdint>

enum MfIndex : uint8_t {
    INDEX_0 = 0U,
    INDEX_1 = 1U,
    INDEX_2 = 2U,
    INDEX_3 = 3U,
    INDEX_4 = 4U,
    INDEX_5 = 5U,
    INDEX_6 = 6U,
};

namespace shm {
namespace utils {
template <typename T>
struct IsUnsignedNumber {
    static constexpr bool value =
        std::is_same<T, unsigned short>::value ||
        std::is_same<T, unsigned int>::value ||
        std::is_same<T, unsigned long>::value ||
        std::is_same<T, unsigned long long>::value;
};

class NumUtil {
public:
    /**
     * @brief Check whether an arithmetic operation will overflow
     *
     * checks potential overflow in addition and multiplication
     *
     * @tparam T      Numeric type (integral)
     * @param a       [in] first operand
     * @param b       [in] second operand
     * @param calc    [in] operation type: '+' for addition, '*' for multiplication
     * @return true if the operation overflow, false otherwise
     */
    template <typename T>
    static bool IsOverflowCheck(T a, T b, T max, char calc);

    /**
     * @brief Check whether the input string is all digits
     *
     * @param str input string
     * @return true if the input is all digits else false
     */
    static bool IsDigit(const std::string& str);

    /**
    * Extracts a bit field from the given flag value.
    *
    * @param flag        The 32-bit source value.
    * @param startBit    The starting bit position (0-based, 0 = least significant bit).
    * @param bitLength   The number of bits to extract (1 to 32).
    * @return            extraction flags value
    */
    static uint32_t ExtractBits(uint32_t flag, uint8_t startBit, uint8_t bitLength)
    {
        if (startBit >= UINT32_WIDTH) {
            return UINT32_MAX;
        }
        if (bitLength == 0 || bitLength >= UINT32_WIDTH) {
            return UINT32_MAX;
        }
        if (startBit + bitLength > UINT32_WIDTH) {
            return UINT32_MAX;
        }

        return (flag >> startBit) & ((1U << bitLength) - 1);
    }
};

template <typename T>
inline bool NumUtil::IsOverflowCheck(T a, T b, T max, char calc)
{
    if (!(IsUnsignedNumber<T>::value)) {
        return false;
    }
    switch (calc) {
        case '+':
            return (a > max - b);
        case '*':
            return ((b != 0) && (a > max / b));
        default:
            return true;
    }
}

inline bool NumUtil::IsDigit(const std::string& str)
{
    if (str.empty()) {
        return false;
    }
    size_t start = str.find_first_not_of(" \t");
    if (start == std::string::npos) {
        return false;
    }

    for (size_t i = start; i < str.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(str[i]))) {
            return false;
        }
    }
    return true;
}
}
}
#endif