/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MTE_PERFTEST_COMMON_H
#define MTE_PERFTEST_COMMON_H

#include <algorithm>  
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <getopt.h>
#include <fstream>

#include "perftest_common_types.h"

#define FOR_EACH_DATATYPE(op) \
    op(float) \
    op(int8_t) \
    op(int16_t) \
    op(int32_t) \
    op(int64_t) \
    op(uint8_t) \
    op(uint16_t) \
    op(uint32_t) \
    op(uint64_t) \
    op(char)

#define DISPATCH_BY_TYPE(type_name, type_op) \
    do { \
        if (strcmp(type_name, "float") == 0) { type_op(float); } \
        else if (strcmp(type_name, "int8") == 0) { type_op(int8_t); } \
        else if (strcmp(type_name, "int16") == 0) { type_op(int16_t); } \
        else if (strcmp(type_name, "int32") == 0) { type_op(int32_t); } \
        else if (strcmp(type_name, "int64") == 0) { type_op(int64_t); } \
        else if (strcmp(type_name, "uint8") == 0) { type_op(uint8_t); } \
        else if (strcmp(type_name, "uint16") == 0) { type_op(uint16_t); } \
        else if (strcmp(type_name, "uint32") == 0) { type_op(uint32_t); } \
        else if (strcmp(type_name, "uint64") == 0) { type_op(uint64_t); } \
        else if (strcmp(type_name, "char") == 0) { type_op(char); } \
    } while(0)

inline perftest::perf_data_type_t get_data_type(const char *data_type_str) {
    if (strcmp(data_type_str, "float") == 0) return perftest::DATA_TYPE_FLOAT;
    else if (strcmp(data_type_str, "int8") == 0) return perftest::DATA_TYPE_INT8;
    else if (strcmp(data_type_str, "int16") == 0) return perftest::DATA_TYPE_INT16;
    else if (strcmp(data_type_str, "int32") == 0) return perftest::DATA_TYPE_INT32;
    else if (strcmp(data_type_str, "int64") == 0) return perftest::DATA_TYPE_INT64;
    else if (strcmp(data_type_str, "uint8") == 0) return perftest::DATA_TYPE_UINT8;
    else if (strcmp(data_type_str, "uint16") == 0) return perftest::DATA_TYPE_UINT16;
    else if (strcmp(data_type_str, "uint32") == 0) return perftest::DATA_TYPE_UINT32;
    else if (strcmp(data_type_str, "uint64") == 0) return perftest::DATA_TYPE_UINT64;
    else if (strcmp(data_type_str, "char") == 0) return perftest::DATA_TYPE_CHAR;
    return perftest::DATA_TYPE_FLOAT;
}

inline size_t get_datatype_size(const char *data_type) {
    if (strcmp(data_type, "float") == 0) return sizeof(float);
    else if (strcmp(data_type, "int8") == 0) return sizeof(int8_t);
    else if (strcmp(data_type, "int16") == 0) return sizeof(int16_t);
    else if (strcmp(data_type, "int32") == 0) return sizeof(int32_t);
    else if (strcmp(data_type, "int64") == 0) return sizeof(int64_t);
    else if (strcmp(data_type, "uint8") == 0) return sizeof(uint8_t);
    else if (strcmp(data_type, "uint16") == 0) return sizeof(uint16_t);
    else if (strcmp(data_type, "uint32") == 0) return sizeof(uint32_t);
    else if (strcmp(data_type, "uint64") == 0) return sizeof(uint64_t);
    else if (strcmp(data_type, "char") == 0) return sizeof(char);
    else return sizeof(float);
}

#endif
