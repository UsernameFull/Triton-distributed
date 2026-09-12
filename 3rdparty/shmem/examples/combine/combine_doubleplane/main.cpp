/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

#include "acl/acl.h"
#include "opdev/fp16_t.h"
#include "param.h"
#include "shmem.h"
#include "utils.h"

#include "combine_kernel.h"

using fp16_t = op::fp16_t;

namespace {

constexpr int MAGIC_MULTIPLIER = 1024;
constexpr int ASSIST_FIELDS = 3;
constexpr size_t ALIGN_BYTES = 32;
constexpr int FULL_FRAME_ID = 0;
constexpr int COMM_FRAME_ID = 1;
constexpr int COMBINE_UB_SIZE_KB = 190;

int g_npus = 8;
const char *ipport = "tcp://127.0.0.1:8766";
int f_npu = 0;
const char *data_type = "int32_t";
aclshmemx_uniqueid_t default_flag_uid;
aclshmem_prof_pe_t *out_profs = nullptr;

struct CombineArgs {
    int pe_size = 2;
    int pe_id = 0;
    int bs = 8;
    int h = 7168;
    int topk = 8;
    int expert_per_pe = 2;
};

struct PerfArgs {
    int perf_mode = 0;
    int warmup_count = 0;
    int loop_count = 1;
    std::string csv_path;
    std::string case_id;
};

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

template <typename T>
bool CheckArray(const T *actual, const T *expected, size_t count, int pe_id)
{
    for (size_t i = 0; i < count; ++i) {
        if (!AlmostEqual(actual[i], expected[i])) {
            std::cerr << "[Combine] x_out mismatch, pe_id=" << pe_id << ", idx=" << i
                      << ", actual=" << static_cast<float>(actual[i])
                      << ", expected=" << static_cast<float>(expected[i]) << std::endl;
            return false;
        }
    }
    return true;
}

size_t AlignUp(size_t value, size_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

size_t CombineWindowBytes(size_t bs, size_t h, size_t topk, size_t elem_size)
{
    const size_t payload_stride = AlignUp(h * elem_size, ALIGN_BYTES);
    return bs * topk * payload_stride + bs * topk * ALIGN_BYTES;
}

bool MakeDirs(const std::string &path)
{
    if (path.empty()) {
        return true;
    }
    std::string current;
    for (size_t i = 0; i < path.size(); ++i) {
        current.push_back(path[i]);
        if (path[i] != '/' && i + 1 != path.size()) {
            continue;
        }
        if (current.empty() || current == "/") {
            continue;
        }
        if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
            return false;
        }
    }
    return true;
}

int GetCycleToUs()
{
    const char *soc_name = aclrtGetSocName();
    if (soc_name != nullptr && std::string(soc_name).find("Ascend950") != std::string::npos) {
        return 1000;
    }
    return 50;
}

double BytesToGBps(size_t bytes, double us)
{
    if (us <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(bytes) / us * 1000000.0 / 1024.0 / 1024.0 / 1024.0;
}

std::string DoubleToString(double value)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << value;
    return oss.str();
}

int GetProfPe()
{
    const char *prof_pe_env = std::getenv("SHMEM_CYCLE_PROF_PE");
    if (prof_pe_env == nullptr) {
        return -1;
    }
    return std::atoi(prof_pe_env);
}

std::vector<double> GetCoreTimesUs(aclshmem_prof_pe_t *profs, int frame_id, int block_num, double *max_time)
{
    std::vector<double> core_times;
    *max_time = 0.0;
    if (profs == nullptr || frame_id < 0 || frame_id >= ACLSHMEM_CYCLE_PROF_FRAME_CNT) {
        return core_times;
    }

    const int actual_blocks = std::min(block_num, ACLSHMEM_CYCLE_PROF_MAX_BLOCK);
    const int cycle_to_us = GetCycleToUs();
    for (int block_id = 0; block_id < actual_blocks; ++block_id) {
        aclshmem_prof_block_t *prof = &profs->block_prof[block_id];
        if (prof->ccount[frame_id] == 0) {
            continue;
        }
        const double avg_us = static_cast<double>(prof->cycles[frame_id]) /
                              static_cast<double>(prof->ccount[frame_id]) / static_cast<double>(cycle_to_us);
        *max_time = std::max(*max_time, avg_us);
        core_times.push_back(avg_us);
    }
    return core_times;
}

void AppendPerfCsvRows(const std::string &csv_path, const CombineArgs &args, const PerfArgs &perf_args,
                       size_t elem_size, int block_num)
{
    if (csv_path.empty()) {
        return;
    }

    aclshmemx_show_prof(&out_profs, false);
    if (out_profs == nullptr) {
        return;
    }

    const bool file_exists = (access(csv_path.c_str(), F_OK) == 0);
    const size_t per_pe_bytes = static_cast<size_t>(args.bs) * args.topk * args.h * elem_size;
    const size_t global_bytes = per_pe_bytes * args.pe_size;
    const int prof_pe = GetProfPe();

    std::string dir = get_dir(csv_path);
    if (!dir.empty()) {
        MakeDirs(dir);
    }

    std::ofstream out_file(csv_path, std::ios::app);
    if (!out_file.is_open()) {
        std::cerr << "[Combine] failed to open perf csv: " << csv_path << std::endl;
        return;
    }

    if (!file_exists) {
        out_file << "DataSize/B,Npus,Blocks,UBsize/KB,Bandwidth/GB/s,CoreMaxTime/us,Metric,GlobalDataSize/B,"
                    "PerPeBandwidth/GB/s,BS,H,TopK,ExpertPerPe,Dtype,Warmup,Loops,ProfPe,CaseId,"
                    "SingleCoreTime/us\n";
    }

    const auto write_row = [&](const std::string &metric, int frame_id) {
        double max_time = 0.0;
        std::vector<double> core_times = GetCoreTimesUs(out_profs, frame_id, block_num, &max_time);
        out_file << per_pe_bytes << "," << args.pe_size << "," << block_num << "," << COMBINE_UB_SIZE_KB << ","
                 << DoubleToString(BytesToGBps(global_bytes, max_time)) << "," << DoubleToString(max_time) << ","
                 << metric << "," << global_bytes << "," << DoubleToString(BytesToGBps(per_pe_bytes, max_time))
                 << "," << args.bs << "," << args.h << "," << args.topk << "," << args.expert_per_pe << ","
                 << data_type << "," << perf_args.warmup_count << "," << perf_args.loop_count << "," << prof_pe
                 << "," << perf_args.case_id;
        for (double core_time : core_times) {
            out_file << "," << DoubleToString(core_time);
        }
        out_file << "\n";
    };

    write_row("full_op", FULL_FRAME_ID);
    write_row("comm_only", COMM_FRAME_ID);
}

template <class T>
int RunCombineCase(const CombineArgs &args, const PerfArgs &perf_args)
{
    if (args.expert_per_pe <= 0 || args.pe_size <= 0 || args.pe_id < 0 || args.pe_id >= args.pe_size ||
        args.bs <= 0 || args.h <= 0 || args.topk <= 0 || g_npus <= 0) {
        std::cerr << "[Combine] invalid arguments." << std::endl;
        return 1;
    }

    aclrtStream stream = nullptr;
    T *expand_x_host = nullptr;
    int32_t *assist_host = nullptr;
    int32_t *ep_recv_count_host = nullptr;
    int32_t *expert_ids_host = nullptr;
    float *expert_scales_host = nullptr;
    T *x_out_host = nullptr;
    T *golden_x_out = nullptr;
    void *expand_x_device = nullptr;
    void *assist_device = nullptr;
    void *ep_recv_count_device = nullptr;
    void *expert_ids_device = nullptr;
    void *expert_scales_device = nullptr;
    void *x_out_device = nullptr;
    void *shmem_window = nullptr;

    auto cleanup = [&]() {
        if (shmem_window != nullptr) {
            aclshmem_free(shmem_window);
            shmem_window = nullptr;
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
        if (expert_ids_device != nullptr) {
            ACL_CHECK(aclrtFree(expert_ids_device));
            expert_ids_device = nullptr;
        }
        if (expert_scales_device != nullptr) {
            ACL_CHECK(aclrtFree(expert_scales_device));
            expert_scales_device = nullptr;
        }
        if (x_out_device != nullptr) {
            ACL_CHECK(aclrtFree(x_out_device));
            x_out_device = nullptr;
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
        if (expert_ids_host != nullptr) {
            ACL_CHECK(aclrtFreeHost(expert_ids_host));
            expert_ids_host = nullptr;
        }
        if (expert_scales_host != nullptr) {
            ACL_CHECK(aclrtFreeHost(expert_scales_host));
            expert_scales_host = nullptr;
        }
        if (x_out_host != nullptr) {
            ACL_CHECK(aclrtFreeHost(x_out_host));
            x_out_host = nullptr;
        }
        if (golden_x_out != nullptr) {
            ACL_CHECK(aclrtFreeHost(golden_x_out));
            golden_x_out = nullptr;
        }
        if (stream != nullptr) {
            ACL_CHECK(aclrtDestroyStream(stream));
            stream = nullptr;
        }
    };
    auto check_acl = [&](aclError error, const char *op_name) -> bool {
        if (error != ACL_ERROR_NONE) {
            std::cerr << "[Combine] " << op_name << " failed, aclError=" << error << std::endl;
            return false;
        }
        return true;
    };
    auto check_not_null = [&](const void *ptr, const char *name) -> bool {
        if (ptr == nullptr) {
            std::cerr << "[Combine] " << name << " returned nullptr." << std::endl;
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

    const int64_t moe_expert_num = static_cast<int64_t>(args.pe_size) * args.expert_per_pe;
    const int64_t max_recv_tokens = static_cast<int64_t>(args.pe_size) * args.bs * args.topk;
    const int64_t segment_num = static_cast<int64_t>(args.expert_per_pe) * args.pe_size;

    const std::string case_dir = "golden/shape_" + std::to_string(args.bs) + "_" +
                                 std::to_string(args.h) + "_" + std::to_string(args.topk) + "_" +
                                 std::to_string(moe_expert_num) + "_" + std::to_string(args.pe_size);
    const std::string rank_dir = case_dir + "/rank_" + std::to_string(args.pe_id);

    const size_t expand_x_bytes = static_cast<size_t>(max_recv_tokens) * args.h * sizeof(T);
    const size_t assist_bytes = static_cast<size_t>(max_recv_tokens) * ASSIST_FIELDS * sizeof(int32_t);
    const size_t ep_recv_count_bytes = static_cast<size_t>(segment_num) * sizeof(int32_t);
    const size_t expert_ids_bytes = static_cast<size_t>(args.bs) * args.topk * sizeof(int32_t);
    const size_t expert_scales_bytes = static_cast<size_t>(args.bs) * args.topk * sizeof(float);
    const size_t x_out_bytes = static_cast<size_t>(args.bs) * args.h * sizeof(T);
    const size_t shmem_window_bytes = CombineWindowBytes(args.bs, args.h, args.topk, sizeof(T));

    if (!alloc_host(reinterpret_cast<void **>(&expand_x_host), expand_x_bytes, "aclrtMallocHost(expand_x_host)") ||
        !alloc_host(reinterpret_cast<void **>(&assist_host), assist_bytes, "aclrtMallocHost(assist_host)") ||
        !alloc_host(reinterpret_cast<void **>(&ep_recv_count_host), ep_recv_count_bytes,
                    "aclrtMallocHost(ep_recv_count_host)") ||
        !alloc_host(reinterpret_cast<void **>(&expert_ids_host), expert_ids_bytes, "aclrtMallocHost(expert_ids_host)") ||
        !alloc_host(reinterpret_cast<void **>(&expert_scales_host), expert_scales_bytes,
                    "aclrtMallocHost(expert_scales_host)")) {
        cleanup();
        return 1;
    }

    if (!ReadFile(rank_dir + "/expand_x.bin", expand_x_host, expand_x_bytes) ||
        !ReadFile(rank_dir + "/assist_info.bin", assist_host, assist_bytes) ||
        !ReadFile(rank_dir + "/ep_recv_count.bin", ep_recv_count_host, ep_recv_count_bytes) ||
        !ReadFile(rank_dir + "/expert_ids.bin", expert_ids_host, expert_ids_bytes) ||
        !ReadFile(rank_dir + "/expert_scales.bin", expert_scales_host, expert_scales_bytes)) {
        std::cerr << "[Combine] failed to read input files from " << rank_dir << std::endl;
        cleanup();
        return 1;
    }

    if (!alloc_device(&expand_x_device, expand_x_bytes, "aclrtMalloc(expand_x_device)") ||
        !alloc_device(&assist_device, assist_bytes, "aclrtMalloc(assist_device)") ||
        !alloc_device(&ep_recv_count_device, ep_recv_count_bytes, "aclrtMalloc(ep_recv_count_device)") ||
        !alloc_device(&expert_ids_device, expert_ids_bytes, "aclrtMalloc(expert_ids_device)") ||
        !alloc_device(&expert_scales_device, expert_scales_bytes, "aclrtMalloc(expert_scales_device)") ||
        !alloc_device(&x_out_device, x_out_bytes, "aclrtMalloc(x_out_device)")) {
        cleanup();
        return 1;
    }
    shmem_window = aclshmem_malloc(shmem_window_bytes);
    if (!check_not_null(shmem_window, "aclshmem_malloc(shmem_window)")) {
        cleanup();
        return 1;
    }

    if (!check_acl(aclrtMemcpy(expand_x_device, expand_x_bytes, expand_x_host, expand_x_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                   "aclrtMemcpy(expand_x_device)") ||
        !check_acl(aclrtMemcpy(assist_device, assist_bytes, assist_host, assist_bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                   "aclrtMemcpy(assist_device)") ||
        !check_acl(aclrtMemcpy(ep_recv_count_device, ep_recv_count_bytes, ep_recv_count_host, ep_recv_count_bytes,
                               ACL_MEMCPY_HOST_TO_DEVICE),
                   "aclrtMemcpy(ep_recv_count_device)") ||
        !check_acl(aclrtMemcpy(expert_ids_device, expert_ids_bytes, expert_ids_host, expert_ids_bytes,
                               ACL_MEMCPY_HOST_TO_DEVICE),
                   "aclrtMemcpy(expert_ids_device)") ||
        !check_acl(aclrtMemcpy(expert_scales_device, expert_scales_bytes, expert_scales_host, expert_scales_bytes,
                               ACL_MEMCPY_HOST_TO_DEVICE),
                   "aclrtMemcpy(expert_scales_device)") ||
        !check_acl(aclrtMemset(x_out_device, x_out_bytes, 0, x_out_bytes), "aclrtMemset(x_out_device)") ||
        !check_acl(aclrtMemset(shmem_window, shmem_window_bytes, 0, shmem_window_bytes), "aclrtMemset(shmem_window)") ||
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
        combine_demo<T>(block_num, stream, fftsAddr, reinterpret_cast<uint8_t *>(expand_x_device),
                        reinterpret_cast<int32_t *>(assist_device), reinterpret_cast<int32_t *>(ep_recv_count_device),
                        reinterpret_cast<int32_t *>(expert_ids_device), reinterpret_cast<float *>(expert_scales_device),
                        reinterpret_cast<uint8_t *>(x_out_device), reinterpret_cast<uint8_t *>(shmem_window), args.bs,
                        args.h, args.topk, moe_expert_num, MAGIC_MULTIPLIER, launch_perf_mode, FULL_FRAME_ID,
                        COMM_FRAME_ID, 0, 1);

        if (!check_acl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
            cleanup();
            return 1;
        }
        aclshmem_barrier_all();
    }

    if (!alloc_host(reinterpret_cast<void **>(&x_out_host), x_out_bytes, "aclrtMallocHost(x_out_host)") ||
        !alloc_host(reinterpret_cast<void **>(&golden_x_out), x_out_bytes, "aclrtMallocHost(golden_x_out)")) {
        cleanup();
        return 1;
    }
    if (!check_acl(aclrtMemcpy(x_out_host, x_out_bytes, x_out_device, x_out_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
                   "aclrtMemcpy(x_out_host)")) {
        cleanup();
        return 1;
    }
    if (!ReadFile(rank_dir + "/golden_x_out.bin", golden_x_out, x_out_bytes)) {
        std::cerr << "[Combine] failed to read golden output from " << rank_dir << std::endl;
        cleanup();
        return 1;
    }

    make_dir("output");
    const std::string output_file = "output/x_out_" + std::to_string(args.pe_id) + ".bin";
    unlink(output_file.c_str());
    WriteFile(output_file, x_out_host, x_out_bytes);

    const bool ok = CheckArray(x_out_host, golden_x_out, static_cast<size_t>(args.bs) * args.h, args.pe_id);

    if (ok && perf_args.perf_mode && args.pe_id == GetProfPe()) {
        AppendPerfCsvRows(perf_args.csv_path, args, perf_args, sizeof(T), static_cast<int>(block_num));
    }

    cleanup();

    if (!ok) {
        return 1;
    }

    std::cout << "[Combine] pe " << args.pe_id << " result correct." << std::endl;
    return 0;
}

} // namespace

int main(int argc, char *argv[])
{
    if (argc < 12) {
        std::cerr << "Usage: combine pe_size pe_id ipport g_npus first_pe first_npu data_type bs h topk "
                     "expert_per_pe"
                  << std::endl;
        return 1;
    }

    CombineArgs args;
    args.pe_size = std::atoi(argv[INDEX1]);
    args.pe_id = std::atoi(argv[INDEX2]);
    ipport = argv[INDEX3];
    g_npus = std::atoi(argv[INDEX4]);
    f_npu = std::atoi(argv[INDEX6]);
    data_type = argv[INDEX7];
    args.bs = std::atoi(argv[INDEX8]);
    args.h = std::atoi(argv[INDEX9]);
    args.topk = std::atoi(argv[10]);
    args.expert_per_pe = std::atoi(argv[11]);

    PerfArgs perf_args;
    if (argc > 12) {
        perf_args.perf_mode = std::atoi(argv[12]);
    }
    if (argc > 13) {
        perf_args.warmup_count = std::atoi(argv[13]);
    }
    if (argc > 14) {
        perf_args.loop_count = std::atoi(argv[14]);
    }
    if (argc > 15) {
        perf_args.csv_path = argv[15];
    }
    if (argc > 16) {
        perf_args.case_id = argv[16];
    }
    if (perf_args.case_id.empty()) {
        perf_args.case_id = "shape_" + std::to_string(args.bs) + "_" + std::to_string(args.h) + "_" +
                            std::to_string(args.topk) + "_" +
                            std::to_string(args.pe_size * args.expert_per_pe) + "_" +
                            std::to_string(args.pe_size);
    }
    if (perf_args.warmup_count < 0) {
        perf_args.warmup_count = 0;
    }
    if (perf_args.loop_count <= 0) {
        perf_args.loop_count = 1;
    }
    if (perf_args.perf_mode && std::getenv("SHMEM_CYCLE_PROF_PE") == nullptr) {
        setenv("SHMEM_CYCLE_PROF_PE", "0", 0);
    }

    const int32_t device_id = args.pe_id % g_npus + f_npu;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(device_id));

    aclshmemx_init_attr_t attributes;
    test_set_attr(args.pe_id, args.pe_size, 1024UL * 1024UL * 1024, ipport, default_flag_uid, &attributes);
    attributes.option_attr.data_op_engine_type = static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_SDMA);
    ACL_CHECK(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes));

    int status = 0;
    if (std::string(data_type) == "int" || std::string(data_type) == "int32_t") {
        status = RunCombineCase<int32_t>(args, perf_args);
    } else if (std::string(data_type) == "float16_t") {
        status = RunCombineCase<fp16_t>(args, perf_args);
    } else {
        std::cerr << "[Combine] unsupported data type: " << data_type << std::endl;
        status = 1;
    }

    ACL_CHECK(aclshmem_finalize());
    ACL_CHECK(aclrtResetDevice(device_id));
    ACL_CHECK(aclFinalize());

    if (status != 0) {
        return status;
    }
    std::cout << "[SUCCESS] combine demo run success in pe " << args.pe_id << std::endl;
    return 0;
}
