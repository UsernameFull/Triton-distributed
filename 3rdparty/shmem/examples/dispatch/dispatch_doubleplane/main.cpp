/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>

#include "acl/acl.h"
#include "opdev/bfloat16.h"
#include "opdev/fp16_t.h"
#include "param.h"
#include "shmem.h"
#include "utils.h"
#include "moe_perf_host.h"

#include "dispatch_kernel.h"

using fp16_t = op::fp16_t;
using bfloat16 = op::bfloat16;

namespace {

constexpr int MAGIC_MULTIPLIER = 1024;
constexpr int ASSIST_FIELDS = 3;
constexpr size_t ALIGN_BYTES = 32;

int g_npus = 8;
const char *ipport = "tcp://127.0.0.1:8766";
int f_npu = 0;
const char *data_type = "int32_t";
aclshmemx_uniqueid_t default_flag_uid;
aclshmem_prof_pe_t *out_profs = nullptr;

constexpr int FULL_FRAME_ID = 0;
constexpr int COMM_FRAME_ID = 1;
constexpr int DISPATCH_UB_SIZE_KB = 190;

template <typename T>
bool AlmostEqual(T lhs, T rhs)
{
    return static_cast<float>(lhs) == static_cast<float>(rhs);
}

template <>
bool AlmostEqual<fp16_t>(fp16_t lhs, fp16_t rhs)
{
    const float diff = std::abs(static_cast<float>(lhs) - static_cast<float>(rhs));
    return diff <= 1.0e-2F;
}

template <>
bool AlmostEqual<bfloat16>(bfloat16 lhs, bfloat16 rhs)
{
    const float diff = std::abs(static_cast<float>(lhs) - static_cast<float>(rhs));
    return diff <= 1.0e-2F;
}

template <typename T>
bool CheckArray(const std::string &name, const T *actual, const T *expected, size_t count, int pe_id)
{
    for (size_t i = 0; i < count; ++i) {
        if (!AlmostEqual(actual[i], expected[i])) {
            std::cerr << "[DispatchDoubleplane] " << name << " mismatch, pe_id=" << pe_id << ", idx=" << i
                      << ", actual=" << static_cast<float>(actual[i])
                      << ", expected=" << static_cast<float>(expected[i]) << std::endl;
            return false;
        }
    }
    return true;
}

bool CheckIntArray(const std::string &name, const int32_t *actual, const int32_t *expected, size_t count, int pe_id)
{
    for (size_t i = 0; i < count; ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << "[DispatchDoubleplane] " << name << " mismatch, pe_id=" << pe_id << ", idx=" << i
                      << ", actual=" << actual[i] << ", expected=" << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

size_t AlignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

size_t DispatchWindowBytes(size_t bs, size_t h, size_t topk, size_t pe_size, size_t local_expert_num, size_t elem_size)
{
    const size_t max_tokens_per_segment = bs * topk;
    const size_t segment_num = pe_size * local_expert_num;
    const size_t total_slots = segment_num * max_tokens_per_segment;
    const size_t payload_stride = AlignUp(h * elem_size, ALIGN_BYTES);
    return total_slots * payload_stride +
           total_slots * ALIGN_BYTES +
           total_slots * ALIGN_BYTES +
           segment_num * ALIGN_BYTES +
           segment_num * ALIGN_BYTES;
}

size_t DispatchCountOffset(size_t bs, size_t h, size_t topk, size_t pe_size, size_t local_expert_num,
                           size_t elem_size)
{
    const size_t max_tokens_per_segment = bs * topk;
    const size_t segment_num = pe_size * local_expert_num;
    const size_t total_slots = segment_num * max_tokens_per_segment;
    const size_t payload_stride = AlignUp(h * elem_size, ALIGN_BYTES);
    return total_slots * payload_stride + total_slots * ALIGN_BYTES + total_slots * ALIGN_BYTES;
}

size_t DispatchCountBytes(size_t pe_size, size_t local_expert_num)
{
    return pe_size * local_expert_num * ALIGN_BYTES;
}

template <class T>
int RunDispatchCase(const MoeShapeArgs &args, const MoePerfArgs &perf_args)
{
    if (args.expert_per_pe <= 0 || args.pe_size <= 0 || args.pe_id < 0 || args.pe_id >= args.pe_size ||
        args.bs <= 0 || args.h <= 0 || args.topk <= 0 || g_npus <= 0) {
        std::cerr << "[DispatchDoubleplane] invalid arguments." << std::endl;
        return 1;
    }
    if (args.expert_per_pe > DISPATCH_MAX_LOCAL_EXPERT_NUM) {
        std::cerr << "[DispatchDoubleplane] invalid expert_per_pe=" << args.expert_per_pe
                  << ", max supported value is " << DISPATCH_MAX_LOCAL_EXPERT_NUM << "." << std::endl;
        return 1;
    }

    aclrtStream stream = nullptr;
    T *x_host = nullptr;
    int32_t *expert_ids_host = nullptr;
    T *expand_x_host = nullptr;
    int32_t *assist_host = nullptr;
    int32_t *ep_recv_count_host = nullptr;
    int32_t *expert_token_nums_host = nullptr;
    T *golden_expand_x = nullptr;
    int32_t *golden_assist = nullptr;
    int32_t *golden_ep_recv_count = nullptr;
    int32_t *golden_expert_token_nums = nullptr;
    void *x_device = nullptr;
    void *expert_ids_device = nullptr;
    void *expand_x_device = nullptr;
    void *assist_device = nullptr;
    void *ep_recv_count_device = nullptr;
    void *expert_token_nums_device = nullptr;
    void *shmem_window = nullptr;

    auto cleanup = [&]() {
        if (shmem_window != nullptr) {
            aclshmem_free(shmem_window);
            shmem_window = nullptr;
        }
        if (x_device != nullptr) {
            ACL_CHECK(aclrtFree(x_device));
            x_device = nullptr;
        }
        if (expert_ids_device != nullptr) {
            ACL_CHECK(aclrtFree(expert_ids_device));
            expert_ids_device = nullptr;
        }
        if (expand_x_device != nullptr) {
            ACL_CHECK(aclrtFree(expand_x_device));
            expand_x_device = nullptr;
        }
        if (assist_device != nullptr) {
            ACL_CHECK(aclrtFree(assist_device));
            assist_device = nullptr;
        }
        if (ep_recv_count_device != nullptr) {
            ACL_CHECK(aclrtFree(ep_recv_count_device));
            ep_recv_count_device = nullptr;
        }
        if (expert_token_nums_device != nullptr) {
            ACL_CHECK(aclrtFree(expert_token_nums_device));
            expert_token_nums_device = nullptr;
        }
        if (x_host != nullptr) {
            ACL_CHECK(aclrtFreeHost(x_host));
            x_host = nullptr;
        }
        if (expert_ids_host != nullptr) {
            ACL_CHECK(aclrtFreeHost(expert_ids_host));
            expert_ids_host = nullptr;
        }
        if (expand_x_host != nullptr) {
            ACL_CHECK(aclrtFreeHost(expand_x_host));
            expand_x_host = nullptr;
        }
        if (assist_host != nullptr) {
            ACL_CHECK(aclrtFreeHost(assist_host));
            assist_host = nullptr;
        }
        if (ep_recv_count_host != nullptr) {
            ACL_CHECK(aclrtFreeHost(ep_recv_count_host));
            ep_recv_count_host = nullptr;
        }
        if (expert_token_nums_host != nullptr) {
            ACL_CHECK(aclrtFreeHost(expert_token_nums_host));
            expert_token_nums_host = nullptr;
        }
        if (golden_expand_x != nullptr) {
            ACL_CHECK(aclrtFreeHost(golden_expand_x));
            golden_expand_x = nullptr;
        }
        if (golden_assist != nullptr) {
            ACL_CHECK(aclrtFreeHost(golden_assist));
            golden_assist = nullptr;
        }
        if (golden_ep_recv_count != nullptr) {
            ACL_CHECK(aclrtFreeHost(golden_ep_recv_count));
            golden_ep_recv_count = nullptr;
        }
        if (golden_expert_token_nums != nullptr) {
            ACL_CHECK(aclrtFreeHost(golden_expert_token_nums));
            golden_expert_token_nums = nullptr;
        }
        if (stream != nullptr) {
            ACL_CHECK(aclrtDestroyStream(stream));
            stream = nullptr;
        }
    };
    auto check_acl = [&](aclError error, const char *op_name) -> bool {
        if (error != ACL_ERROR_NONE) {
            std::cerr << "[DispatchDoubleplane] " << op_name << " failed, aclError=" << error << std::endl;
            return false;
        }
        return true;
    };
    auto check_not_null = [&](const void *ptr, const char *name) -> bool {
        if (ptr == nullptr) {
            std::cerr << "[DispatchDoubleplane] " << name << " returned nullptr." << std::endl;
            return false;
        }
        return true;
    };
    auto alloc_host = [&](void **ptr, size_t bytes, const char *name) -> bool {
        if (!check_acl(aclrtMallocHost(ptr, bytes), name)) {
            return false;
        }
        return check_not_null(*ptr, name);
    };
    auto alloc_device = [&](void **ptr, size_t bytes, const char *name) -> bool {
        if (!check_acl(aclrtMalloc(ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), name)) {
            return false;
        }
        return check_not_null(*ptr, name);
    };

    if (!check_acl(aclrtCreateStream(&stream), "aclrtCreateStream") || !check_not_null(stream, "aclrtCreateStream")) {
        cleanup();
        return 1;
    }
    const uint64_t fftsAddr = util_get_ffts_config();

    const int64_t moe_expert_num = static_cast<int64_t>(args.expert_per_pe) * args.pe_size;
    const int64_t max_recv_tokens = static_cast<int64_t>(args.pe_size) * args.bs * args.topk;
    const int64_t segment_num = static_cast<int64_t>(args.expert_per_pe) * args.pe_size;

    const std::string case_dir = "golden/shape_" + std::to_string(args.bs) + "_" +
                                 std::to_string(args.h) + "_" + std::to_string(args.topk) + "_" +
                                 std::to_string(moe_expert_num) + "_" + std::to_string(args.pe_size);
    const std::string rank_dir = case_dir + "/rank_" + std::to_string(args.pe_id);

    const size_t x_bytes = static_cast<size_t>(args.bs) * args.h * sizeof(T);
    const size_t expert_ids_bytes = static_cast<size_t>(args.bs) * args.topk * sizeof(int32_t);
    const size_t expand_x_bytes = static_cast<size_t>(max_recv_tokens) * args.h * sizeof(T);
    const size_t assist_bytes = static_cast<size_t>(max_recv_tokens) * ASSIST_FIELDS * sizeof(int32_t);
    const size_t ep_recv_count_bytes = static_cast<size_t>(segment_num) * sizeof(int32_t);
    const size_t expert_token_nums_bytes = static_cast<size_t>(args.expert_per_pe) * sizeof(int32_t);
    const size_t shmem_window_bytes =
        DispatchWindowBytes(args.bs, args.h, args.topk, args.pe_size, args.expert_per_pe, sizeof(T));
    const size_t shmem_count_offset =
        DispatchCountOffset(args.bs, args.h, args.topk, args.pe_size, args.expert_per_pe, sizeof(T));
    const size_t shmem_count_bytes = DispatchCountBytes(args.pe_size, args.expert_per_pe);

    if (!alloc_host(reinterpret_cast<void **>(&x_host), x_bytes, "aclrtMallocHost(x_host)") ||
        !alloc_host(reinterpret_cast<void **>(&expert_ids_host), expert_ids_bytes, "aclrtMallocHost(expert_ids_host)")) {
        cleanup();
        return 1;
    }
    if (!ReadFile(rank_dir + "/x.bin", x_host, x_bytes) ||
        !ReadFile(rank_dir + "/expert_ids.bin", expert_ids_host, expert_ids_bytes)) {
        std::cerr << "[DispatchDoubleplane] failed to read input files from " << rank_dir << std::endl;
        cleanup();
        return 1;
    }

    if (!alloc_device(&x_device, x_bytes, "aclrtMalloc(x_device)") ||
        !alloc_device(&expert_ids_device, expert_ids_bytes, "aclrtMalloc(expert_ids_device)") ||
        !alloc_device(&expand_x_device, expand_x_bytes, "aclrtMalloc(expand_x_device)") ||
        !alloc_device(&assist_device, assist_bytes, "aclrtMalloc(assist_device)") ||
        !alloc_device(&ep_recv_count_device, ep_recv_count_bytes, "aclrtMalloc(ep_recv_count_device)") ||
        !alloc_device(&expert_token_nums_device, expert_token_nums_bytes, "aclrtMalloc(expert_token_nums_device)")) {
        cleanup();
        return 1;
    }
    shmem_window = aclshmem_malloc(shmem_window_bytes);
    if (!check_not_null(shmem_window, "aclshmem_malloc(shmem_window)")) {
        cleanup();
        return 1;
    }

    if (!check_acl(aclrtMemcpy(x_device, x_bytes, x_host, x_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                   "aclrtMemcpy(x_device)") ||
        !check_acl(aclrtMemcpy(expert_ids_device, expert_ids_bytes, expert_ids_host, expert_ids_bytes,
                               ACL_MEMCPY_HOST_TO_DEVICE),
                   "aclrtMemcpy(expert_ids_device)") ||
        !check_acl(aclrtMemset(expand_x_device, expand_x_bytes, 0, expand_x_bytes), "aclrtMemset(expand_x_device)") ||
        !check_acl(aclrtMemset(assist_device, assist_bytes, 0, assist_bytes), "aclrtMemset(assist_device)") ||
        !check_acl(aclrtMemset(ep_recv_count_device, ep_recv_count_bytes, 0, ep_recv_count_bytes),
                   "aclrtMemset(ep_recv_count_device)") ||
        !check_acl(aclrtMemset(expert_token_nums_device, expert_token_nums_bytes, 0, expert_token_nums_bytes),
                   "aclrtMemset(expert_token_nums_device)") ||
        !check_acl(aclrtMemset(shmem_window, shmem_window_bytes, 0, shmem_window_bytes), "aclrtMemset(shmem_window)") ||
        !check_acl(aclrtMemset(reinterpret_cast<uint8_t *>(shmem_window) + shmem_count_offset, shmem_count_bytes,
                               0xFF, shmem_count_bytes),
                   "aclrtMemset(shmem_count_window)") ||
        !check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
        cleanup();
        return 1;
    }
    aclshmem_barrier_all();

    const uint32_t block_num = static_cast<uint32_t>(args.pe_size);
    const int warmup_count = perf_args.perf_mode ? perf_args.warmup_count : 0;
    const int loop_count = perf_args.perf_mode ? perf_args.loop_count : 1;
    for (int iter = 0; iter < warmup_count + loop_count; ++iter) {
        const int launch_perf_mode = (perf_args.perf_mode && iter >= warmup_count) ? 1 : 0;
        dispatch_demo<T>(block_num, stream, fftsAddr, reinterpret_cast<uint8_t *>(x_device),
                         reinterpret_cast<int32_t *>(expert_ids_device), reinterpret_cast<uint8_t *>(expand_x_device),
                         reinterpret_cast<int32_t *>(assist_device), reinterpret_cast<int32_t *>(ep_recv_count_device),
                         reinterpret_cast<int32_t *>(expert_token_nums_device),
                         reinterpret_cast<uint8_t *>(shmem_window), args.bs, args.h, args.topk, moe_expert_num,
                         MAGIC_MULTIPLIER, launch_perf_mode, FULL_FRAME_ID, COMM_FRAME_ID, 0, 1);

        if (!check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
            cleanup();
            return 1;
        }
        aclshmem_barrier_all();
    }

    if (!alloc_host(reinterpret_cast<void **>(&expand_x_host), expand_x_bytes, "aclrtMallocHost(expand_x_host)") ||
        !alloc_host(reinterpret_cast<void **>(&assist_host), assist_bytes, "aclrtMallocHost(assist_host)") ||
        !alloc_host(reinterpret_cast<void **>(&ep_recv_count_host), ep_recv_count_bytes,
                    "aclrtMallocHost(ep_recv_count_host)") ||
        !alloc_host(reinterpret_cast<void **>(&expert_token_nums_host), expert_token_nums_bytes,
                    "aclrtMallocHost(expert_token_nums_host)")) {
        cleanup();
        return 1;
    }

    if (!check_acl(aclrtMemcpy(expand_x_host, expand_x_bytes, expand_x_device, expand_x_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                   "aclrtMemcpy(expand_x_host)") ||
        !check_acl(aclrtMemcpy(assist_host, assist_bytes, assist_device, assist_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                   "aclrtMemcpy(assist_host)") ||
        !check_acl(aclrtMemcpy(ep_recv_count_host, ep_recv_count_bytes, ep_recv_count_device, ep_recv_count_bytes,
                               ACL_MEMCPY_DEVICE_TO_HOST),
                   "aclrtMemcpy(ep_recv_count_host)") ||
        !check_acl(aclrtMemcpy(expert_token_nums_host, expert_token_nums_bytes, expert_token_nums_device,
                               expert_token_nums_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                   "aclrtMemcpy(expert_token_nums_host)")) {
        cleanup();
        return 1;
    }

    make_dir("output");
    const std::string out_prefix = "output/";
    const std::string pe_suffix = std::to_string(args.pe_id) + ".bin";
    const std::string expand_out = out_prefix + "expand_x_" + pe_suffix;
    const std::string assist_out = out_prefix + "assist_info_" + pe_suffix;
    const std::string recv_out = out_prefix + "ep_recv_count_" + pe_suffix;
    const std::string expert_out = out_prefix + "expert_token_nums_" + pe_suffix;
    unlink(expand_out.c_str());
    unlink(assist_out.c_str());
    unlink(recv_out.c_str());
    unlink(expert_out.c_str());
    WriteFile(expand_out, expand_x_host, expand_x_bytes);
    WriteFile(assist_out, assist_host, assist_bytes);
    WriteFile(recv_out, ep_recv_count_host, ep_recv_count_bytes);
    WriteFile(expert_out, expert_token_nums_host, expert_token_nums_bytes);

    if (!alloc_host(reinterpret_cast<void **>(&golden_expand_x), expand_x_bytes, "aclrtMallocHost(golden_expand_x)") ||
        !alloc_host(reinterpret_cast<void **>(&golden_assist), assist_bytes, "aclrtMallocHost(golden_assist)") ||
        !alloc_host(reinterpret_cast<void **>(&golden_ep_recv_count), ep_recv_count_bytes,
                    "aclrtMallocHost(golden_ep_recv_count)") ||
        !alloc_host(reinterpret_cast<void **>(&golden_expert_token_nums), expert_token_nums_bytes,
                    "aclrtMallocHost(golden_expert_token_nums)")) {
        cleanup();
        return 1;
    }

    if (!ReadFile(rank_dir + "/golden_expand_x.bin", golden_expand_x, expand_x_bytes) ||
        !ReadFile(rank_dir + "/golden_assist_info.bin", golden_assist, assist_bytes) ||
        !ReadFile(rank_dir + "/golden_ep_recv_count.bin", golden_ep_recv_count, ep_recv_count_bytes) ||
        !ReadFile(rank_dir + "/golden_expert_token_nums.bin", golden_expert_token_nums, expert_token_nums_bytes)) {
        std::cerr << "[DispatchDoubleplane] failed to read golden files from " << rank_dir << std::endl;
        cleanup();
        return 1;
    }

    const bool ok = CheckArray("expand_x", expand_x_host, golden_expand_x, max_recv_tokens * args.h, args.pe_id) &&
                    CheckIntArray("assist_info", assist_host, golden_assist, max_recv_tokens * ASSIST_FIELDS,
                                  args.pe_id) &&
                    CheckIntArray("ep_recv_count", ep_recv_count_host, golden_ep_recv_count, segment_num,
                                  args.pe_id) &&
                    CheckIntArray("expert_token_nums", expert_token_nums_host, golden_expert_token_nums,
                                  args.expert_per_pe, args.pe_id);

    if (ok && perf_args.perf_mode && args.pe_id == MoeGetProfPe()) {
        MoeAppendPerfCsvRows("DispatchDoubleplane", &out_profs, perf_args.csv_path, args, perf_args, data_type, sizeof(T),
                             static_cast<int>(block_num), DISPATCH_UB_SIZE_KB);
    }

    cleanup();

    if (!ok) {
        return 1;
    }

    std::cout << "[DispatchDoubleplane] pe " << args.pe_id << " result correct." << std::endl;
    return 0;
}

} // namespace

int main(int argc, char *argv[])
{
    if (argc < INDEX12) {
        std::cerr << "Usage: dispatch_doubleplane pe_size pe_id ipport g_npus first_pe first_npu data_type bs h topk "
                     "expert_per_pe"
                  << std::endl;
        return 1;
    }

    MoeShapeArgs args;
    args.pe_size = std::atoi(argv[INDEX1]);
    args.pe_id = std::atoi(argv[INDEX2]);
    ipport = argv[INDEX3];
    g_npus = std::atoi(argv[INDEX4]);
    // Keep the `first_pe` CLI slot for compatibility with shared example scripts.
    const int first_pe = std::atoi(argv[INDEX5]);
    (void)first_pe;
    f_npu = std::atoi(argv[INDEX6]);
    data_type = argv[INDEX7];
    args.bs = std::atoi(argv[INDEX8]);
    args.h = std::atoi(argv[INDEX9]);
    args.topk = std::atoi(argv[INDEX10]);
    args.expert_per_pe = std::atoi(argv[INDEX11]);

    MoePerfArgs perf_args;
    MoeInitPerfArgsFromArgv(perf_args, argc, argv, args, INDEX12);

    const int32_t device_id = args.pe_id % g_npus + f_npu;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(device_id));

    aclshmemx_init_attr_t attributes;
    test_set_attr(args.pe_id, args.pe_size, 1024UL * 1024UL * 1024, ipport, default_flag_uid, &attributes);
    attributes.option_attr.data_op_engine_type = static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_SDMA);
    ACL_CHECK(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

    int status = 0;
    if (std::string(data_type) == "int" || std::string(data_type) == "int32_t") {
        status = RunDispatchCase<int32_t>(args, perf_args);
    } else if (std::string(data_type) == "float16_t") {
        status = RunDispatchCase<fp16_t>(args, perf_args);
    } else if (std::string(data_type) == "bfloat16_t") {
        status = RunDispatchCase<bfloat16>(args, perf_args);
    } else {
        std::cerr << "[DispatchDoubleplane] unsupported data type: " << data_type << std::endl;
        status = 1;
    }

    ACL_CHECK(aclshmem_finalize());
    ACL_CHECK(aclrtResetDevice(device_id));
    ACL_CHECK(aclFinalize());

    if (status != 0) {
        return status;
    }
    std::cout << "[SUCCESS] dispatch_doubleplane demo run success in pe " << args.pe_id << std::endl;
    return 0;
}
