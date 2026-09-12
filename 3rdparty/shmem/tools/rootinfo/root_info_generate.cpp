/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "topo_addr_info.h"

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <phyId>" << std::endl;
        return 1;
    }

    int phyId = std::atoi(argv[1]);
    std::cout << "Generating root info for NPU with physical ID: " << phyId << std::endl;

    std::size_t size = 0;
    int ret = topo_addr_info_get_size(phyId, &size);
    if (ret != 0) {
        std::cerr << "Error: topo_addr_info_get_size failed with ret=" << ret << std::endl;
        return 1;
    }
    std::cout << "Required buffer size: " << size << " bytes" << std::endl;

    std::vector<char> rank_info(size);
    std::size_t actual_size = size;
    ret = topo_addr_info_get(phyId, rank_info.data(), &actual_size);
    if (ret != 0) {
        std::cerr << "Error: topo_addr_info_get failed with ret=" << ret << std::endl;
        return 1;
    }
    std::cout << "topo_addr_info_get succeeded, actual size: " << actual_size << " bytes" << std::endl;
    std::cout << "Rank info:" << std::endl << std::string(rank_info.data(), actual_size) << std::endl;

    std::vector<char> file_path(256);
    ret = topo_addr_info_get_topo_file_path(phyId, file_path.data(), file_path.size());
    if (ret != 0) {
        std::cerr << "Error: topo_addr_info_get_topo_file_path failed with ret=" << ret << std::endl;
        return 1;
    }
    std::cout << "Topology file path: " << std::string(file_path.data()) << std::endl;

    std::cout << "Root info generation completed successfully" << std::endl;
    return 0;
}