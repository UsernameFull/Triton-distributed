/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MOE_PERF_HOST_H
#define MOE_PERF_HOST_H

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
#include "host_device/shmem_common_types.h"
#include "shmem.h"
#include "utils.h"

using MoeString = std::string;
using MoeDoubleVector = std::vector<double>;
using MoeOutputFileStream = std::ofstream;

struct MoeShapeArgs {
    int pe_size = 2;
    int pe_id = 0;
    int bs = 8;
    int h = 16;
    int topk = 2;
    int expert_per_pe = 2;
};

struct MoePerfArgs {
    int perf_mode = 0;
    int warmup_count = 0;
    int loop_count = 1;
    std::string csv_path;
    std::string case_id;
};

inline bool MoeMakeDirs(const std::string &path)
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

inline int MoeGetCycleToUs()
{
    const char *soc_name = aclrtGetSocName();
    if (soc_name != nullptr && std::string(soc_name).find("Ascend950") != std::string::npos) {
        return 1000;
    }
    return 50;
}

inline double MoeBytesToGBps(size_t bytes, double us)
{
    if (us <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(bytes) / us * 1000000.0 / 1024.0 / 1024.0 / 1024.0;
}

inline std::string MoeDoubleToString(double value)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << value;
    return oss.str();
}

inline int MoeGetProfPe()
{
    const char *prof_pe_env = std::getenv("SHMEM_CYCLE_PROF_PE");
    if (prof_pe_env == nullptr) {
        return -1;
    }
    return std::atoi(prof_pe_env);
}

inline void MoeInitPerfArgsFromArgv(MoePerfArgs &perf_args, int argc, char *argv[], const MoeShapeArgs &shape_args,
                                    int first_optional_idx)
{
    if (argc > first_optional_idx) {
        perf_args.perf_mode = std::atoi(argv[first_optional_idx]);
    }
    if (argc > first_optional_idx + 1) {
        perf_args.warmup_count = std::atoi(argv[first_optional_idx + 1]);
    }
    if (argc > first_optional_idx + 2) {
        perf_args.loop_count = std::atoi(argv[first_optional_idx + 2]);
    }
    if (argc > first_optional_idx + 3) {
        perf_args.csv_path = argv[first_optional_idx + 3];
    }
    if (argc > first_optional_idx + 4) {
        perf_args.case_id = argv[first_optional_idx + 4];
    }
    if (perf_args.case_id.empty()) {
        perf_args.case_id = "shape_" + std::to_string(shape_args.bs) + "_" + std::to_string(shape_args.h) + "_" +
                            std::to_string(shape_args.topk) + "_" +
                            std::to_string(shape_args.pe_size * shape_args.expert_per_pe) + "_" +
                            std::to_string(shape_args.pe_size);
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
}

inline std::vector<double> MoeGetCoreTimesUs(aclshmem_prof_pe_t *profs, int frame_id, int block_num,
                                             double *max_time)
{
    std::vector<double> core_times;
    *max_time = 0.0;
    if (profs == nullptr || frame_id < 0 || frame_id >= ACLSHMEM_CYCLE_PROF_FRAME_CNT) {
        return core_times;
    }

    const int actual_blocks = std::min(block_num, ACLSHMEM_CYCLE_PROF_MAX_BLOCK);
    const int cycle_to_us = MoeGetCycleToUs();
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

inline void MoeAppendPerfCsvRows(const MoeString &label, aclshmem_prof_pe_t **out_profs,
                                 const MoeString &csv_path, const MoeShapeArgs &shape_args,
                                 const MoePerfArgs &perf_args, const char *dtype, size_t elem_size, int block_num,
                                 int ub_size_kb, int full_frame_id = 0, int comm_frame_id = 1)
{
    if (csv_path.empty()) {
        return;
    }

    aclshmemx_show_prof(out_profs, false);
    if (out_profs == nullptr || *out_profs == nullptr) {
        return;
    }

    const bool file_exists = (access(csv_path.c_str(), F_OK) == 0);
    const size_t per_pe_bytes = static_cast<size_t>(shape_args.bs) * shape_args.topk * shape_args.h * elem_size;
    const size_t global_bytes = per_pe_bytes * shape_args.pe_size;
    const int prof_pe = MoeGetProfPe();

    MoeString dir = get_dir(csv_path);
    if (!dir.empty()) {
        MoeMakeDirs(dir);
    }

    MoeOutputFileStream out_file(csv_path, std::ios::app);
    if (!out_file.is_open()) {
        std::cerr << "[" << label << "] failed to open perf csv: " << csv_path << std::endl;
        return;
    }

    if (!file_exists) {
        out_file << "DataSize/B,Npus,Blocks,UBsize/KB,Bandwidth/GB/s,CoreMaxTime/us,Metric,GlobalDataSize/B,"
                    "PerPeBandwidth/GB/s,BS,H,TopK,ExpertPerPe,Dtype,Warmup,Loops,ProfPe,CaseId,"
                    "SingleCoreTime/us\n";
    }

    const auto write_row = [&](const MoeString &metric, int frame_id) {
        double max_time = 0.0;
        MoeDoubleVector core_times = MoeGetCoreTimesUs(*out_profs, frame_id, block_num, &max_time);
        out_file << per_pe_bytes << "," << shape_args.pe_size << "," << block_num << "," << ub_size_kb << ","
                 << MoeDoubleToString(MoeBytesToGBps(global_bytes, max_time)) << ","
                 << MoeDoubleToString(max_time) << "," << metric << "," << global_bytes << ","
                 << MoeDoubleToString(MoeBytesToGBps(per_pe_bytes, max_time)) << "," << shape_args.bs << ","
                 << shape_args.h << "," << shape_args.topk << "," << shape_args.expert_per_pe << "," << dtype << ","
                 << perf_args.warmup_count << "," << perf_args.loop_count << "," << prof_pe << ","
                 << perf_args.case_id;
        for (double core_time : core_times) {
            out_file << "," << MoeDoubleToString(core_time);
        }
        out_file << "\n";
    };

    write_row("full_op", full_frame_id);
    write_row("comm_only", comm_frame_id);
}

#endif // MOE_PERF_HOST_H
