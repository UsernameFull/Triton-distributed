/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "pcie_nic.h"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <arpa/inet.h>
#include <dirent.h>
#include <ifaddrs.h>
#include <libgen.h>
#include <net/if.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <securec.h>

#include "aclshmemi_hal.h"
#include "utils/shmemi_logger.h"

constexpr int MAX_NIC_PATH = 256 + 32;
constexpr int MAX_NIC_COUNT = 32;
constexpr int PCIE_ROOT_MIN_LEN = 23;
constexpr int MAX_PCIE_BUS_ID_LEN = 32;

typedef struct {
    char path[MAX_NIC_PATH];
    char pcie_path[MAX_NIC_PATH];
    size_t comon_prefix_len;
} nic;

typedef struct _stNPU {
    int id = 0;
    char bus_id[MAX_PCIE_BUS_ID_LEN] = {0};
    char path[MAX_NIC_PATH] = {0};
    char nic_path[MAX_NIC_PATH] = {0};
    char nic_name[MAX_NIC_PATH] = {0};
} NPU;

static size_t get_common_prefix_len(const char* a, const char* b)
{
    size_t len = 0;
    while (a[len] == b[len] && a[len] != '\0' && b[len] != '\0') {
        len++;
    }
    return len;
}

static int get_abs_path(const char* path, char* abs_path, size_t abs_path_len)
{
    char resolved_path[PATH_MAX] = {0};
    char* p = realpath(path, resolved_path);
    if (p == NULL) {
        return -1;
    }
    return strcpy_s(abs_path, abs_path_len, resolved_path);
}

int get_nics(nic* nics, size_t* nic_len)
{
    DIR* dir = opendir("/sys/class/net");
    if (dir == NULL) {
        return -1;
    }

    int loop = 0;
    struct dirent* entry = NULL;
    while ((entry = readdir(dir)) != NULL) {
        if (loop >= MAX_NIC_COUNT) {
            break;
        }
        if (entry->d_name[0] == '.') {
            continue;
        }
        if (strcmp(entry->d_name, "lo") == 0) {
            continue;
        }

        int ret = sprintf_s(nics[loop].path, MAX_NIC_PATH, "/sys/class/net/%s", entry->d_name);
        if (ret < 0) {
            closedir(dir);
            SHM_LOG_ERROR("get_nics failed: sprintf_s failed for NIC " << entry->d_name);
            return -1;
        }

        get_abs_path(nics[loop].path, nics[loop].pcie_path, MAX_NIC_PATH);
        loop++;
    }

    *nic_len = loop;
    closedir(dir);
    return 0;
}

static int get_pcie_nics(NPU* npu, size_t pos, size_t npu_count)
{
    char dev_path[MAX_NIC_PATH] = {0};
    char abs_path[MAX_NIC_PATH] = {0};
    size_t abs_path_len = sizeof(abs_path);
    int ret = sprintf_s(dev_path, sizeof(dev_path), "/sys/bus/pci/devices/%s", npu[pos].bus_id);
    if (ret < 0) {
        return -1;
    }
    get_abs_path(dev_path, abs_path, abs_path_len);

    nic nics[MAX_NIC_COUNT];
    memset_s(nics, sizeof(nics), 0, sizeof(nics));
    size_t nic_len = MAX_NIC_COUNT;
    ret = get_nics(nics, &nic_len);
    if (ret != 0) {
        return -1;
    }

    int max_pos = -1;
    size_t maxlen = PCIE_ROOT_MIN_LEN;
    for (size_t i = 0; i < nic_len; ++i) {
        size_t common_prefix_len = get_common_prefix_len(abs_path, nics[i].pcie_path);
        if (common_prefix_len > maxlen) {
            int skip = 0;
            for (size_t j = 0; j < npu_count; ++j) {
                if (strcmp(nics[i].pcie_path, npu[j].nic_path) == 0) {
                    skip = 1;
                    break;
                }
            }
            if (skip) {
                continue;
            }
            max_pos = i;
            maxlen = common_prefix_len;
        }
    }
    if (max_pos < 0) {
        return -1;
    }
    (void)strcpy_s(npu[pos].nic_path, sizeof(npu[pos].nic_path), nics[max_pos].pcie_path);
    (void)strcpy_s(npu[pos].nic_name, sizeof(npu[pos].nic_name), basename(nics[max_pos].pcie_path));
    return 0;
}

static void init_npus(NPU* npu, size_t npu_count)
{
    memset_s(npu, sizeof(NPU) * npu_count, 0, sizeof(NPU) * npu_count);
    auto& hal = shm::topo::aclshmemi_hal_t::instance();

    for (size_t i = 0; i < npu_count; ++i) {
        npu[i].id = i;

        struct dcmi_pcie_info_all pcie_info;
        memset_s(&pcie_info, sizeof(pcie_info), 0, sizeof(pcie_info));
        int ret = hal.get_device_pcie_info(i, &pcie_info);
        if (ret != 0) {
            SHM_LOG_DEBUG("init_npus: get_device_pcie_info failed for npu_id=" << i << ", skipping");
            continue;
        }

        ret = sprintf_s(
            npu[i].bus_id, MAX_PCIE_BUS_ID_LEN, "%04x:%02x:%02x.%x", pcie_info.domain, pcie_info.bdf_busid,
            pcie_info.bdf_deviceid, pcie_info.bdf_funcid);
        if (ret < 0) {
            SHM_LOG_ERROR("init_npus failed: sprintf_s bus_id failed for npu_id=" << i);
            continue;
        }
    }
}

int get_ip_by_if_name(const char* ifname, char* ip, size_t addr_len)
{
    struct ifaddrs* ifaddr = NULL;
    if (getifaddrs(&ifaddr) == -1) {
        return -1;
    }
    int found_interface = 0;
    for (struct ifaddrs* ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || !ifa->ifa_name) {
            continue;
        }
        if (strcmp(ifname, ifa->ifa_name) != 0) {
            continue;
        }
        found_interface = 1;
        if (ifa->ifa_addr->sa_family == AF_INET) {
            struct sockaddr_in* sin = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
            if (inet_ntop(AF_INET, &sin->sin_addr, ip, addr_len) != NULL) {
                freeifaddrs(ifaddr);
                return 0;
            }
            freeifaddrs(ifaddr);
            return -1;
        }
    }
    freeifaddrs(ifaddr);
    errno = found_interface ? ENOENT : ENODEV;
    return -1;
}

int get_npu_roce_ip(int npu_id, char* ip_addr, size_t ip_addr_len)
{
    auto& hal = shm::topo::aclshmemi_hal_t::instance();
    auto npu_count_opt = hal.get_npu_count();
    if (!npu_count_opt) {
        return -1;
    }

    int npu_count = *npu_count_opt;

    // 边界校验：防止负值或超大值导致内存过度分配
    if (npu_count <= 0 || npu_count > ACLSHMEMI_MAX_NPU_COUNT) {
        SHM_LOG_ERROR(
            "get_npu_roce_ip failed: invalid npu_count=" << npu_count << ", expected range [1, "
                                                         << ACLSHMEMI_MAX_NPU_COUNT << "]");
        return -1;
    }

    std::vector<NPU> npus(npu_count);
    init_npus(npus.data(), npu_count);

    for (int i = 0; i < npu_count; ++i) {
        int ret = get_pcie_nics(npus.data(), i, npu_count);
        if (ret != 0) {
            SHM_LOG_DEBUG("get_npu_roce_ip: get_pcie_nics failed for npu_id=" << i);
            continue;
        }
    }

    for (int i = 0; i < npu_count; ++i) {
        if (npus[i].id == npu_id) {
            return get_ip_by_if_name(npus[i].nic_name, ip_addr, ip_addr_len);
        }
    }

    errno = ENODEV;
    return -1;
}