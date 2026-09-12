/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIMT_RMA_GM2GM_DEFINITIONS_H
#define SIMT_RMA_GM2GM_DEFINITIONS_H

#if defined(USE_SIMT)

#include <cstdint>
#include <iostream>

#include "acl/acl.h"

#define TYPED_FUNCTION_DEF(FUNC) \
    FUNC(half) \
    FUNC(float) \
    FUNC(int8) \
    FUNC(int16) \
    FUNC(int32) \
    FUNC(int64) \
    FUNC(uint8) \
    FUNC(uint16) \
    FUNC(uint32) \
    FUNC(uint64) \
    FUNC(char) \
    FUNC(schar) \
    FUNC(bfloat16)

#define TYPE_FUNCTION_TEMPLATE(TYPE)                                                                                            \
    extern void test_aclshmem_##TYPE##_put(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);             \
    extern void test_aclshmemx_##TYPE##_put_warp(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);       \
    extern void test_aclshmemx_##TYPE##_put_block(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);      \
    extern void test_aclshmem_##TYPE##_put_nbi(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);         \
    extern void test_aclshmemx_##TYPE##_put_nbi_warp(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);   \
    extern void test_aclshmemx_##TYPE##_put_nbi_block(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);  \
    extern void test_aclshmem_##TYPE##_get(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);             \
    extern void test_aclshmemx_##TYPE##_get_warp(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);       \
    extern void test_aclshmemx_##TYPE##_get_block(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);      \
    extern void test_aclshmem_##TYPE##_get_nbi(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);         \
    extern void test_aclshmemx_##TYPE##_get_nbi_warp(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);   \
    extern void test_aclshmemx_##TYPE##_get_nbi_block(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);

TYPED_FUNCTION_DEF(TYPE_FUNCTION_TEMPLATE)

#define SCALAR_FUNCTION_TEMPLATE(TYPE) \
    extern void test_aclshmem_##TYPE##_p(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe); \
    extern void test_aclshmem_##TYPE##_g(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);

TYPED_FUNCTION_DEF(SCALAR_FUNCTION_TEMPLATE)

#undef TYPE_FUNCTION_TEMPLATE
#undef SCALAR_FUNCTION_TEMPLATE
#undef TYPED_FUNCTION_DEF

#define SIZED_FUNCTION_DEF(FUNC) \
    FUNC(8) \
    FUNC(16) \
    FUNC(32) \
    FUNC(64) \
    FUNC(128)

#define SIZED_FUNCTION_TEMPLATE(SIZE) \
    extern void test_aclshmem_put##SIZE(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);                \
    extern void test_aclshmemx_put##SIZE##_warp(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);        \
    extern void test_aclshmemx_put##SIZE##_block(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);       \
    extern void test_aclshmem_put##SIZE##_nbi(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);          \
    extern void test_aclshmemx_put##SIZE##_nbi_warp(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);    \
    extern void test_aclshmemx_put##SIZE##_nbi_block(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);   \
    extern void test_aclshmem_get##SIZE(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);                \
    extern void test_aclshmemx_get##SIZE##_warp(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);        \
    extern void test_aclshmemx_get##SIZE##_block(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);       \
    extern void test_aclshmem_get##SIZE##_nbi(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);          \
    extern void test_aclshmemx_get##SIZE##_nbi_warp(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);    \
    extern void test_aclshmemx_get##SIZE##_nbi_block(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);

SIZED_FUNCTION_DEF(SIZED_FUNCTION_TEMPLATE)

#undef SIZED_FUNCTION_DEF
#undef SIZED_FUNCTION_TEMPLATE

extern void test_aclshmem_putmem(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_aclshmemx_putmem_warp(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_aclshmemx_putmem_block(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_aclshmem_putmem_nbi(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_aclshmemx_putmem_nbi_warp(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_aclshmemx_putmem_nbi_block(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_aclshmem_getmem(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_aclshmemx_getmem_warp(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_aclshmemx_getmem_block(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_aclshmem_getmem_nbi(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_aclshmemx_getmem_nbi_warp(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);
extern void test_aclshmemx_getmem_nbi_block(aclrtStream stream, void* dst, void* src, size_t elem_size, int32_t pe);


using FunctionInterface = void(aclrtStream, void*, void*, size_t, int32_t);

enum class TypeName {
    HALF, FLOAT, INT8, INT16, INT32, INT64, 
    UINT8, UINT16, UINT32, UINT64, 
    UCHAR, SCHAR, BFLOAT16
};

inline std::ostream& operator<<(std::ostream& os, const TypeName& type) {
    switch (type) {
        case TypeName::HALF: os << "HALF"; break;
        case TypeName::FLOAT: os << "FLOAT"; break;
        case TypeName::INT8: os << "INT8"; break;
        case TypeName::INT16: os << "INT16"; break;
        case TypeName::INT32: os << "INT32"; break;
        case TypeName::INT64: os << "INT64"; break;
        case TypeName::UINT8: os << "UINT8"; break;
        case TypeName::UINT16: os << "UINT16"; break;
        case TypeName::UINT32: os << "UINT32"; break;
        case TypeName::UINT64: os << "UINT64"; break;
        case TypeName::UCHAR: os << "UCHAR"; break;
        case TypeName::SCHAR: os << "SCHAR"; break;
        case TypeName::BFLOAT16: os << "BFLOAT16"; break;
    }
    return os;
}

enum class Bits {
    BITS8, BITS16, BITS32, BITS64, BITS128
};

inline std::ostream& operator<<(std::ostream& os, const Bits& bits) {
    switch (bits) {
        case Bits::BITS8: os << "BITS8"; break;
        case Bits::BITS16: os << "BITS16"; break;
        case Bits::BITS32: os << "BITS32"; break;
        case Bits::BITS64: os << "BITS64"; break;
        case Bits::BITS128: os << "BITS128"; break;
    }
    return os;
}

enum class OpType {
    PUT, GET
};

inline std::ostream& operator<<(std::ostream& os, const OpType& op) {
    switch (op) {
        case OpType::PUT: os << "PUT"; break;
        case OpType::GET: os << "GET"; break;
    }
    return os;
}

enum class Scope {
    THREAD, WARP, BLOCK
};

inline std::ostream& operator<<(std::ostream& os, const Scope& scope) {
    switch (scope) {
        case Scope::THREAD: os << "THREAD"; break;
        case Scope::WARP: os << "WARP"; break;
        case Scope::BLOCK: os << "BLOCK"; break;
    }
    return os;
}

enum class IOMode {
    BLOCKING, NON_BLOCKING
};

inline std::ostream& operator<<(std::ostream& os, const IOMode& mode) {
    switch (mode) {
        case IOMode::BLOCKING: os << "BLOCKING"; break;
        case IOMode::NON_BLOCKING: os << "NON_BLOCKING"; break;
    }
    return os;
}

enum class SigType {
    Typed, Sized, Memory, Scalar
};

inline std::ostream& operator<<(std::ostream& os, const SigType& sig) {
    switch (sig) {
        case SigType::Typed: os << "Typed"; break;
        case SigType::Sized: os << "Sized"; break;
        case SigType::Memory: os << "Memory"; break;
        case SigType::Scalar: os << "Scalar"; break;
    }
    return os;
}

template <SigType sig, typename... Ts>
class Sig;

template <>
class Sig<SigType::Typed, OpType, TypeName, IOMode, Scope> 
{
public:
    OpType op;
    TypeName name;
    IOMode io_mode;
    Scope scope;

    Sig(OpType op, TypeName name, IOMode io_mode, Scope scope) : op(op), name(name), io_mode(io_mode), scope(scope) {}
};

inline std::ostream& operator<<(std::ostream& os, const Sig<SigType::Typed, OpType, TypeName, IOMode, Scope>& sig) 
{
    os << "Sig<Typed>[" << sig.op << ", " << sig.name << ", " << sig.io_mode << ", " << sig.scope << "]";
    return os;
}

template <>
class Sig<SigType::Sized, OpType, Bits, IOMode, Scope> 
{
public:
    OpType op;
    Bits bits;
    IOMode io_mode;
    Scope scope;

    Sig(OpType op, Bits bits, IOMode io_mode, Scope scope) : op(op), bits(bits), io_mode(io_mode), scope(scope) {}
};

inline std::ostream& operator<<(std::ostream& os, const Sig<SigType::Sized, OpType, Bits, IOMode, Scope>& sig) {
    os << "Sig<Sized>[" << sig.op << ", " << sig.bits << ", " << sig.io_mode << ", " << sig.scope << "]";
    return os;
}

template <>
class Sig<SigType::Memory, OpType, IOMode, Scope> {
public:
    OpType op;
    IOMode io_mode;
    Scope scope;

    Sig(OpType op, IOMode io_mode, Scope scope) : op(op), io_mode(io_mode), scope(scope) {}
};

inline std::ostream& operator<<(std::ostream& os, const Sig<SigType::Memory, OpType, IOMode, Scope>& sig) {
    os << "Sig<Memory>[" << sig.op << ", " << sig.io_mode << ", " << sig.scope << "]";
    return os;
}

template <>
class Sig<SigType::Scalar, OpType, TypeName> {
public:
    OpType op;
    TypeName name;

    Sig(OpType op, TypeName name) : op(op), name(name) {}
};

inline std::ostream& operator<<(std::ostream& os, const Sig<SigType::Scalar, OpType, TypeName>& sig) {
    os << "Sig<Scalar>[" << sig.op << ", " << sig.name << "]";
    return os;
}

using TypedSig = Sig<SigType::Typed, OpType, TypeName, IOMode, Scope>;
using SizedSig = Sig<SigType::Sized, OpType, Bits, IOMode, Scope>;
using MemorySig = Sig<SigType::Memory, OpType, IOMode, Scope>;
using ScalarSig = Sig<SigType::Scalar, OpType, TypeName>;
using Sigs = std::variant<TypedSig, SizedSig, MemorySig, ScalarSig>;

constexpr std::pair<TypeName, size_t> TypeSizeMap[] = {
    {TypeName::HALF, sizeof(int16_t)},
    {TypeName::FLOAT, sizeof(float)},
    {TypeName::INT8, sizeof(int8_t)},
    {TypeName::INT16, sizeof(int16_t)},
    {TypeName::INT32, sizeof(int32_t)},
    {TypeName::INT64, sizeof(int64_t)},
    {TypeName::UINT8, sizeof(uint8_t)},
    {TypeName::UINT16, sizeof(uint16_t)},
    {TypeName::UINT32, sizeof(uint32_t)},
    {TypeName::UINT64, sizeof(uint64_t)},
    {TypeName::UCHAR, sizeof(unsigned char)},
    {TypeName::SCHAR, sizeof(signed char)},
    {TypeName::BFLOAT16, sizeof(int16_t)}
};

size_t get_type_size(TypeName type) {
    for (const auto& pair : TypeSizeMap) {
        if (pair.first == type) {
            return pair.second;
        }
    }
    return 0;
}

constexpr std::pair<Bits, size_t> BitsSizeMap[] = {
    {Bits::BITS8, 1},
    {Bits::BITS16, 2},
    {Bits::BITS32, 4},
    {Bits::BITS64, 8},
    {Bits::BITS128, 16}
};

size_t get_bits_size(Bits bits) {
    for (const auto& pair : BitsSizeMap) {
        if (pair.first == bits) {
            return pair.second;
        }
    }
    return 0;
}

using TestCaseEntry = std::tuple<FunctionInterface*, Sigs>;
static std::vector<TestCaseEntry> Entries = {
    std::make_tuple(test_aclshmem_half_put, TypedSig(OpType::PUT, TypeName::HALF, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_half_put_warp, TypedSig(OpType::PUT, TypeName::HALF, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_half_put_block, TypedSig(OpType::PUT, TypeName::HALF, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_half_put_nbi, TypedSig(OpType::PUT, TypeName::HALF, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_half_put_nbi_warp, TypedSig(OpType::PUT, TypeName::HALF, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_half_put_nbi_block, TypedSig(OpType::PUT, TypeName::HALF, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_float_put, TypedSig(OpType::PUT, TypeName::FLOAT, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_float_put_warp, TypedSig(OpType::PUT, TypeName::FLOAT, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_float_put_block, TypedSig(OpType::PUT, TypeName::FLOAT, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_float_put_nbi, TypedSig(OpType::PUT, TypeName::FLOAT, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_float_put_nbi_warp, TypedSig(OpType::PUT, TypeName::FLOAT, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_float_put_nbi_block, TypedSig(OpType::PUT, TypeName::FLOAT, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int8_put, TypedSig(OpType::PUT, TypeName::INT8, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int8_put_warp, TypedSig(OpType::PUT, TypeName::INT8, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int8_put_block, TypedSig(OpType::PUT, TypeName::INT8, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int8_put_nbi, TypedSig(OpType::PUT, TypeName::INT8, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int8_put_nbi_warp, TypedSig(OpType::PUT, TypeName::INT8, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int8_put_nbi_block, TypedSig(OpType::PUT, TypeName::INT8, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int16_put, TypedSig(OpType::PUT, TypeName::INT16, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int16_put_warp, TypedSig(OpType::PUT, TypeName::INT16, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int16_put_block, TypedSig(OpType::PUT, TypeName::INT16, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int16_put_nbi, TypedSig(OpType::PUT, TypeName::INT16, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int16_put_nbi_warp, TypedSig(OpType::PUT, TypeName::INT16, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int16_put_nbi_block, TypedSig(OpType::PUT, TypeName::INT16, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int32_put, TypedSig(OpType::PUT, TypeName::INT32, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int32_put_warp, TypedSig(OpType::PUT, TypeName::INT32, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int32_put_block, TypedSig(OpType::PUT, TypeName::INT32, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int32_put_nbi, TypedSig(OpType::PUT, TypeName::INT32, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int32_put_nbi_warp, TypedSig(OpType::PUT, TypeName::INT32, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int32_put_nbi_block, TypedSig(OpType::PUT, TypeName::INT32, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int64_put, TypedSig(OpType::PUT, TypeName::INT64, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int64_put_warp, TypedSig(OpType::PUT, TypeName::INT64, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int64_put_block, TypedSig(OpType::PUT, TypeName::INT64, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int64_put_nbi, TypedSig(OpType::PUT, TypeName::INT64, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int64_put_nbi_warp, TypedSig(OpType::PUT, TypeName::INT64, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int64_put_nbi_block, TypedSig(OpType::PUT, TypeName::INT64, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint8_put, TypedSig(OpType::PUT, TypeName::UINT8, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint8_put_warp, TypedSig(OpType::PUT, TypeName::UINT8, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint8_put_block, TypedSig(OpType::PUT, TypeName::UINT8, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint8_put_nbi, TypedSig(OpType::PUT, TypeName::UINT8, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint8_put_nbi_warp, TypedSig(OpType::PUT, TypeName::UINT8, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint8_put_nbi_block, TypedSig(OpType::PUT, TypeName::UINT8, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint16_put, TypedSig(OpType::PUT, TypeName::UINT16, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint16_put_warp, TypedSig(OpType::PUT, TypeName::UINT16, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint16_put_block, TypedSig(OpType::PUT, TypeName::UINT16, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint16_put_nbi, TypedSig(OpType::PUT, TypeName::UINT16, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint16_put_nbi_warp, TypedSig(OpType::PUT, TypeName::UINT16, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint16_put_nbi_block, TypedSig(OpType::PUT, TypeName::UINT16, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint32_put, TypedSig(OpType::PUT, TypeName::UINT32, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint32_put_warp, TypedSig(OpType::PUT, TypeName::UINT32, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint32_put_block, TypedSig(OpType::PUT, TypeName::UINT32, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint32_put_nbi, TypedSig(OpType::PUT, TypeName::UINT32, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint32_put_nbi_warp, TypedSig(OpType::PUT, TypeName::UINT32, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint32_put_nbi_block, TypedSig(OpType::PUT, TypeName::UINT32, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint64_put, TypedSig(OpType::PUT, TypeName::UINT64, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint64_put_warp, TypedSig(OpType::PUT, TypeName::UINT64, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint64_put_block, TypedSig(OpType::PUT, TypeName::UINT64, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint64_put_nbi, TypedSig(OpType::PUT, TypeName::UINT64, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint64_put_nbi_warp, TypedSig(OpType::PUT, TypeName::UINT64, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint64_put_nbi_block, TypedSig(OpType::PUT, TypeName::UINT64, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_char_put, TypedSig(OpType::PUT, TypeName::UCHAR, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_char_put_warp, TypedSig(OpType::PUT, TypeName::UCHAR, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_char_put_block, TypedSig(OpType::PUT, TypeName::UCHAR, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_char_put_nbi, TypedSig(OpType::PUT, TypeName::UCHAR, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_char_put_nbi_warp, TypedSig(OpType::PUT, TypeName::UCHAR, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_char_put_nbi_block, TypedSig(OpType::PUT, TypeName::UCHAR, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_schar_put, TypedSig(OpType::PUT, TypeName::SCHAR, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_schar_put_warp, TypedSig(OpType::PUT, TypeName::SCHAR, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_schar_put_block, TypedSig(OpType::PUT, TypeName::SCHAR, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_schar_put_nbi, TypedSig(OpType::PUT, TypeName::SCHAR, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_schar_put_nbi_warp, TypedSig(OpType::PUT, TypeName::SCHAR, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_schar_put_nbi_block, TypedSig(OpType::PUT, TypeName::SCHAR, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_bfloat16_put, TypedSig(OpType::PUT, TypeName::BFLOAT16, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_bfloat16_put_warp, TypedSig(OpType::PUT, TypeName::BFLOAT16, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_bfloat16_put_block, TypedSig(OpType::PUT, TypeName::BFLOAT16, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_bfloat16_put_nbi, TypedSig(OpType::PUT, TypeName::BFLOAT16, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_bfloat16_put_nbi_warp, TypedSig(OpType::PUT, TypeName::BFLOAT16, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_bfloat16_put_nbi_block, TypedSig(OpType::PUT, TypeName::BFLOAT16, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_half_get, TypedSig(OpType::GET, TypeName::HALF, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_half_get_warp, TypedSig(OpType::GET, TypeName::HALF, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_half_get_block, TypedSig(OpType::GET, TypeName::HALF, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_half_get_nbi, TypedSig(OpType::GET, TypeName::HALF, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_half_get_nbi_warp, TypedSig(OpType::GET, TypeName::HALF, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_half_get_nbi_block, TypedSig(OpType::GET, TypeName::HALF, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_float_get, TypedSig(OpType::GET, TypeName::FLOAT, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_float_get_warp, TypedSig(OpType::GET, TypeName::FLOAT, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_float_get_block, TypedSig(OpType::GET, TypeName::FLOAT, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_float_get_nbi, TypedSig(OpType::GET, TypeName::FLOAT, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_float_get_nbi_warp, TypedSig(OpType::GET, TypeName::FLOAT, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_float_get_nbi_block, TypedSig(OpType::GET, TypeName::FLOAT, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int8_get, TypedSig(OpType::GET, TypeName::INT8, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int8_get_warp, TypedSig(OpType::GET, TypeName::INT8, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int8_get_block, TypedSig(OpType::GET, TypeName::INT8, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int8_get_nbi, TypedSig(OpType::GET, TypeName::INT8, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int8_get_nbi_warp, TypedSig(OpType::GET, TypeName::INT8, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int8_get_nbi_block, TypedSig(OpType::GET, TypeName::INT8, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int16_get, TypedSig(OpType::GET, TypeName::INT16, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int16_get_warp, TypedSig(OpType::GET, TypeName::INT16, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int16_get_block, TypedSig(OpType::GET, TypeName::INT16, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int16_get_nbi, TypedSig(OpType::GET, TypeName::INT16, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int16_get_nbi_warp, TypedSig(OpType::GET, TypeName::INT16, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int16_get_nbi_block, TypedSig(OpType::GET, TypeName::INT16, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int32_get, TypedSig(OpType::GET, TypeName::INT32, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int32_get_warp, TypedSig(OpType::GET, TypeName::INT32, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int32_get_block, TypedSig(OpType::GET, TypeName::INT32, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int32_get_nbi, TypedSig(OpType::GET, TypeName::INT32, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int32_get_nbi_warp, TypedSig(OpType::GET, TypeName::INT32, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int32_get_nbi_block, TypedSig(OpType::GET, TypeName::INT32, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int64_get, TypedSig(OpType::GET, TypeName::INT64, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int64_get_warp, TypedSig(OpType::GET, TypeName::INT64, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int64_get_block, TypedSig(OpType::GET, TypeName::INT64, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_int64_get_nbi, TypedSig(OpType::GET, TypeName::INT64, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_int64_get_nbi_warp, TypedSig(OpType::GET, TypeName::INT64, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_int64_get_nbi_block, TypedSig(OpType::GET, TypeName::INT64, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint8_get, TypedSig(OpType::GET, TypeName::UINT8, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint8_get_warp, TypedSig(OpType::GET, TypeName::UINT8, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint8_get_block, TypedSig(OpType::GET, TypeName::UINT8, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint8_get_nbi, TypedSig(OpType::GET, TypeName::UINT8, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint8_get_nbi_warp, TypedSig(OpType::GET, TypeName::UINT8, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint8_get_nbi_block, TypedSig(OpType::GET, TypeName::UINT8, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint16_get, TypedSig(OpType::GET, TypeName::UINT16, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint16_get_warp, TypedSig(OpType::GET, TypeName::UINT16, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint16_get_block, TypedSig(OpType::GET, TypeName::UINT16, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint16_get_nbi, TypedSig(OpType::GET, TypeName::UINT16, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint16_get_nbi_warp, TypedSig(OpType::GET, TypeName::UINT16, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint16_get_nbi_block, TypedSig(OpType::GET, TypeName::UINT16, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint32_get, TypedSig(OpType::GET, TypeName::UINT32, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint32_get_warp, TypedSig(OpType::GET, TypeName::UINT32, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint32_get_block, TypedSig(OpType::GET, TypeName::UINT32, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint32_get_nbi, TypedSig(OpType::GET, TypeName::UINT32, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint32_get_nbi_warp, TypedSig(OpType::GET, TypeName::UINT32, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint32_get_nbi_block, TypedSig(OpType::GET, TypeName::UINT32, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint64_get, TypedSig(OpType::GET, TypeName::UINT64, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint64_get_warp, TypedSig(OpType::GET, TypeName::UINT64, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint64_get_block, TypedSig(OpType::GET, TypeName::UINT64, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_uint64_get_nbi, TypedSig(OpType::GET, TypeName::UINT64, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_uint64_get_nbi_warp, TypedSig(OpType::GET, TypeName::UINT64, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_uint64_get_nbi_block, TypedSig(OpType::GET, TypeName::UINT64, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_char_get, TypedSig(OpType::GET, TypeName::UCHAR, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_char_get_warp, TypedSig(OpType::GET, TypeName::UCHAR, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_char_get_block, TypedSig(OpType::GET, TypeName::UCHAR, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_char_get_nbi, TypedSig(OpType::GET, TypeName::UCHAR, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_char_get_nbi_warp, TypedSig(OpType::GET, TypeName::UCHAR, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_char_get_nbi_block, TypedSig(OpType::GET, TypeName::UCHAR, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_schar_get, TypedSig(OpType::GET, TypeName::SCHAR, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_schar_get_warp, TypedSig(OpType::GET, TypeName::SCHAR, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_schar_get_block, TypedSig(OpType::GET, TypeName::SCHAR, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_schar_get_nbi, TypedSig(OpType::GET, TypeName::SCHAR, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_schar_get_nbi_warp, TypedSig(OpType::GET, TypeName::SCHAR, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_schar_get_nbi_block, TypedSig(OpType::GET, TypeName::SCHAR, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_bfloat16_get, TypedSig(OpType::GET, TypeName::BFLOAT16, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_bfloat16_get_warp, TypedSig(OpType::GET, TypeName::BFLOAT16, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_bfloat16_get_block, TypedSig(OpType::GET, TypeName::BFLOAT16, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_bfloat16_get_nbi, TypedSig(OpType::GET, TypeName::BFLOAT16, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_bfloat16_get_nbi_warp, TypedSig(OpType::GET, TypeName::BFLOAT16, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_bfloat16_get_nbi_block, TypedSig(OpType::GET, TypeName::BFLOAT16, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_put8, SizedSig(OpType::PUT, Bits::BITS8, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_put8_warp, SizedSig(OpType::PUT, Bits::BITS8, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_put8_block, SizedSig(OpType::PUT, Bits::BITS8, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_put8_nbi, SizedSig(OpType::PUT, Bits::BITS8, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_put8_nbi_warp, SizedSig(OpType::PUT, Bits::BITS8, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_put8_nbi_block, SizedSig(OpType::PUT, Bits::BITS8, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_put16, SizedSig(OpType::PUT, Bits::BITS16, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_put16_warp, SizedSig(OpType::PUT, Bits::BITS16, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_put16_block, SizedSig(OpType::PUT, Bits::BITS16, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_put16_nbi, SizedSig(OpType::PUT, Bits::BITS16, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_put16_nbi_warp, SizedSig(OpType::PUT, Bits::BITS16, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_put16_nbi_block, SizedSig(OpType::PUT, Bits::BITS16, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_put32, SizedSig(OpType::PUT, Bits::BITS32, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_put32_warp, SizedSig(OpType::PUT, Bits::BITS32, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_put32_block, SizedSig(OpType::PUT, Bits::BITS32, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_put32_nbi, SizedSig(OpType::PUT, Bits::BITS32, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_put32_nbi_warp, SizedSig(OpType::PUT, Bits::BITS32, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_put32_nbi_block, SizedSig(OpType::PUT, Bits::BITS32, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_put64, SizedSig(OpType::PUT, Bits::BITS64, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_put64_warp, SizedSig(OpType::PUT, Bits::BITS64, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_put64_block, SizedSig(OpType::PUT, Bits::BITS64, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_put64_nbi, SizedSig(OpType::PUT, Bits::BITS64, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_put64_nbi_warp, SizedSig(OpType::PUT, Bits::BITS64, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_put64_nbi_block, SizedSig(OpType::PUT, Bits::BITS64, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_put128, SizedSig(OpType::PUT, Bits::BITS128, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_put128_warp, SizedSig(OpType::PUT, Bits::BITS128, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_put128_block, SizedSig(OpType::PUT, Bits::BITS128, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_put128_nbi, SizedSig(OpType::PUT, Bits::BITS128, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_put128_nbi_warp, SizedSig(OpType::PUT, Bits::BITS128, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_put128_nbi_block, SizedSig(OpType::PUT, Bits::BITS128, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_get8, SizedSig(OpType::GET, Bits::BITS8, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_get8_warp, SizedSig(OpType::GET, Bits::BITS8, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_get8_block, SizedSig(OpType::GET, Bits::BITS8, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_get8_nbi, SizedSig(OpType::GET, Bits::BITS8, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_get8_nbi_warp, SizedSig(OpType::GET, Bits::BITS8, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_get8_nbi_block, SizedSig(OpType::GET, Bits::BITS8, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_get16, SizedSig(OpType::GET, Bits::BITS16, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_get16_warp, SizedSig(OpType::GET, Bits::BITS16, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_get16_block, SizedSig(OpType::GET, Bits::BITS16, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_get16_nbi, SizedSig(OpType::GET, Bits::BITS16, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_get16_nbi_warp, SizedSig(OpType::GET, Bits::BITS16, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_get16_nbi_block, SizedSig(OpType::GET, Bits::BITS16, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_get32, SizedSig(OpType::GET, Bits::BITS32, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_get32_warp, SizedSig(OpType::GET, Bits::BITS32, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_get32_block, SizedSig(OpType::GET, Bits::BITS32, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_get32_nbi, SizedSig(OpType::GET, Bits::BITS32, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_get32_nbi_warp, SizedSig(OpType::GET, Bits::BITS32, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_get32_nbi_block, SizedSig(OpType::GET, Bits::BITS32, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_get64, SizedSig(OpType::GET, Bits::BITS64, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_get64_warp, SizedSig(OpType::GET, Bits::BITS64, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_get64_block, SizedSig(OpType::GET, Bits::BITS64, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_get64_nbi, SizedSig(OpType::GET, Bits::BITS64, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_get64_nbi_warp, SizedSig(OpType::GET, Bits::BITS64, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_get64_nbi_block, SizedSig(OpType::GET, Bits::BITS64, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_get128, SizedSig(OpType::GET, Bits::BITS128, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_get128_warp, SizedSig(OpType::GET, Bits::BITS128, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_get128_block, SizedSig(OpType::GET, Bits::BITS128, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_get128_nbi, SizedSig(OpType::GET, Bits::BITS128, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_get128_nbi_warp, SizedSig(OpType::GET, Bits::BITS128, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_get128_nbi_block, SizedSig(OpType::GET, Bits::BITS128, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_putmem, MemorySig(OpType::PUT, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_putmem_warp, MemorySig(OpType::PUT, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_putmem_block, MemorySig(OpType::PUT, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_putmem_nbi, MemorySig(OpType::PUT, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_putmem_nbi_warp, MemorySig(OpType::PUT, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_putmem_nbi_block, MemorySig(OpType::PUT, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_getmem, MemorySig(OpType::GET, IOMode::BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_getmem_warp, MemorySig(OpType::GET, IOMode::BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_getmem_block, MemorySig(OpType::GET, IOMode::BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_getmem_nbi, MemorySig(OpType::GET, IOMode::NON_BLOCKING, Scope::THREAD)),
    std::make_tuple(test_aclshmemx_getmem_nbi_warp, MemorySig(OpType::GET, IOMode::NON_BLOCKING, Scope::WARP)),
    std::make_tuple(test_aclshmemx_getmem_nbi_block, MemorySig(OpType::GET, IOMode::NON_BLOCKING, Scope::BLOCK)),
    std::make_tuple(test_aclshmem_half_p, ScalarSig(OpType::PUT, TypeName::HALF)),
    std::make_tuple(test_aclshmem_float_p, ScalarSig(OpType::PUT, TypeName::FLOAT)),
    std::make_tuple(test_aclshmem_int8_p, ScalarSig(OpType::PUT, TypeName::INT8)),
    std::make_tuple(test_aclshmem_int16_p, ScalarSig(OpType::PUT, TypeName::INT16)),
    std::make_tuple(test_aclshmem_int32_p, ScalarSig(OpType::PUT, TypeName::INT32)),
    std::make_tuple(test_aclshmem_int64_p, ScalarSig(OpType::PUT, TypeName::INT64)),
    std::make_tuple(test_aclshmem_uint8_p, ScalarSig(OpType::PUT, TypeName::UINT8)),
    std::make_tuple(test_aclshmem_uint16_p, ScalarSig(OpType::PUT, TypeName::UINT16)),
    std::make_tuple(test_aclshmem_uint32_p, ScalarSig(OpType::PUT, TypeName::UINT32)),
    std::make_tuple(test_aclshmem_uint64_p, ScalarSig(OpType::PUT, TypeName::UINT64)),
    std::make_tuple(test_aclshmem_char_p, ScalarSig(OpType::PUT, TypeName::UCHAR)),
    std::make_tuple(test_aclshmem_schar_p, ScalarSig(OpType::PUT, TypeName::SCHAR)),
    std::make_tuple(test_aclshmem_bfloat16_p, ScalarSig(OpType::PUT, TypeName::BFLOAT16)),
    std::make_tuple(test_aclshmem_half_g, ScalarSig(OpType::GET, TypeName::HALF)),
    std::make_tuple(test_aclshmem_float_g, ScalarSig(OpType::GET, TypeName::FLOAT)),
    std::make_tuple(test_aclshmem_int8_g, ScalarSig(OpType::GET, TypeName::INT8)),
    std::make_tuple(test_aclshmem_int16_g, ScalarSig(OpType::GET, TypeName::INT16)),
    std::make_tuple(test_aclshmem_int32_g, ScalarSig(OpType::GET, TypeName::INT32)),
    std::make_tuple(test_aclshmem_int64_g, ScalarSig(OpType::GET, TypeName::INT64)),
    std::make_tuple(test_aclshmem_uint8_g, ScalarSig(OpType::GET, TypeName::UINT8)),
    std::make_tuple(test_aclshmem_uint16_g, ScalarSig(OpType::GET, TypeName::UINT16)),
    std::make_tuple(test_aclshmem_uint32_g, ScalarSig(OpType::GET, TypeName::UINT32)),
    std::make_tuple(test_aclshmem_uint64_g, ScalarSig(OpType::GET, TypeName::UINT64)),
    std::make_tuple(test_aclshmem_char_g, ScalarSig(OpType::GET, TypeName::UCHAR)),
    std::make_tuple(test_aclshmem_schar_g, ScalarSig(OpType::GET, TypeName::SCHAR)),
    std::make_tuple(test_aclshmem_bfloat16_g, ScalarSig(OpType::GET, TypeName::BFLOAT16))
};

#endif // defined(ACLSHMEM_SIMT_SUPPORT)

#endif // SIMT_RMA_GM2GM_DEFINITIONS_H