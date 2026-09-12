/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <fstream>
#include <string>
#include <cstdint>
#include "shmemi_file_util.h"
#include "dl_comm_def.h"
#include "utils/shmemi_logger.h"

const std::string DRIVER_VER_V5_BEGIN = "V100R001C10SPC001B208";
const std::string DRIVER_VER_V5_END = "V100R001C11B999";
const std::string DRIVER_VER_V4 = "V100R001C23SPC005B219";
const std::string DRIVER_VER_V3 = "V100R001C21B035";
const std::string DRIVER_VER_V2 = "V100R001C19SPC109B220";
const std::string DRIVER_VER_V1 = "V100R001C18B100";

HybmGvaVersion checkVer = HYBM_GVA_UNKNOWN;

HybmGvaVersion HybmGetGvaVersion()
{
    return checkVer;
}

std::string GetDriverVersionPath(const std::string &driverEnvStr, const std::string &keyStr)
{
    std::string driverVersionPath;
    std::string tempPath;
    for (uint32_t i = 0; i < driverEnvStr.length(); ++i) {
        if (driverEnvStr[i] != ':') {
            tempPath += driverEnvStr[i];
        }
        if (driverEnvStr[i] == ':' || i == driverEnvStr.length() - 1) {
            if (!shm::utils::FileUtil::Realpath(tempPath)) {
                tempPath.clear();
                continue;
            }
            auto found = tempPath.find(keyStr);
            if (found == std::string::npos) {
                tempPath.clear();
                continue;
            }
            if (tempPath.length() <= found + keyStr.length() || tempPath[found + keyStr.length()] == '/') {
                driverVersionPath = tempPath.substr(0, found);
                break;
            }
            tempPath.clear();
        }
    }
    return driverVersionPath;
}

std::string LoadDriverVersionInfoFile(const std::string &realName, const std::string &keyStr)
{
    std::string driverVersion;
    char realFile[shm::utils::FileUtil::GetSafePathMax()] = {0};
    if (shm::utils::FileUtil::IsSymlink(realName) || realpath(realName.c_str(), realFile) == nullptr) {
        SHM_LOG_WARN("driver version path is not a valid real path");
        return "";
    }

    std::ifstream infile(realFile, std::ifstream::in);
    if (!infile.is_open()) {
        SHM_LOG_WARN("driver version file does not exist");
        return "";
    }

    std::string line;
    int32_t maxRows = 100;
    while (getline(infile, line)) {
        --maxRows;
        if (maxRows < 0) {
            SHM_LOG_WARN("driver version file content is too long.");
            infile.close();
            return "";
        }
        auto found = line.find(keyStr);
        if (found == 0) {
            uint32_t len = line.length() - keyStr.length();
            driverVersion = line.substr(keyStr.length(), len);
            break;
        }
    }
    infile.close();
    return driverVersion;
}

std::string CastDriverVersion(const std::string &driverEnv)
{
    std::string driverVersionPath = GetDriverVersionPath(driverEnv, "/driver/lib64");
    if (!driverVersionPath.empty()) {
        driverVersionPath += "/driver/version.info";
        std::string driverVersion = LoadDriverVersionInfoFile(driverVersionPath, "Innerversion=");
        return driverVersion;
    } else {
        SHM_LOG_WARN("cannot found version file in :" << driverEnv << ", try local default path.");
        driverVersionPath = "/usr/local/Ascend/driver/version.info";  // try default path
        std::string driverVersion = LoadDriverVersionInfoFile(driverVersionPath, "Innerversion=");
        return driverVersion;
    }
}

int32_t GetValueFromVersion(const std::string &ver, std::string key)
{
    int32_t val = 0;
    auto found = ver.find(key);
    if (found == std::string::npos) {
        return -1;
    }

    std::string tmp;
    found += 1;
    while (found < ver.length() && std::isdigit(ver[found])) {
        tmp += ver[found];
        ++found;
    }

    try {
        val = std::stoi(tmp);
    } catch (...) {
        val = -1;
    }
    return val;
}

static bool DriverVersionCheck(const std::string &ver)
{
    auto libPath = std::getenv("LD_LIBRARY_PATH");
    if (libPath == nullptr) {
        SHM_LOG_ERROR("check driver version failed, Environment LD_LIBRARY_PATH not set.");
        return false;
    }

    std::string readVer = CastDriverVersion(libPath);
    if (readVer.empty()) {
        SHM_LOG_WARN("Read version is empty; inner package skipped, defaulting to the latest version.");
        return true;
    }

    int32_t baseVal = GetValueFromVersion(ver, "V");
    int32_t readVal = GetValueFromVersion(readVer, "V");
    if (baseVal == -1 || readVal == -1 || baseVal != readVal) {
        SHM_LOG_INFO("check driver version failed, V Version not equal");
        return false;
    }

    baseVal = GetValueFromVersion(ver, "R");
    readVal = GetValueFromVersion(readVer, "R");
    if (baseVal == -1 || readVal == -1 || baseVal != readVal) {
        SHM_LOG_INFO("check driver version failed, R Release not equal");
        return false;
    }

    baseVal = GetValueFromVersion(ver, "C");
    readVal = GetValueFromVersion(readVer, "C");
    if (baseVal == -1 || readVal == -1 || readVal < baseVal) {
        SHM_LOG_INFO("check driver version failed, C Customer is too low");
        return false;
    }
    if (readVal > baseVal) {
        return true;
    }

    baseVal = GetValueFromVersion(ver, "SPC");
    readVal = GetValueFromVersion(readVer, "SPC");
    if (baseVal != -1) {
        if (readVal == -1 || readVal < baseVal) {
            SHM_LOG_DEBUG("Driver version mismatch, base=" << baseVal << ", read=" << readVal);
            return false;
        }
        if (readVal > baseVal) {
            return true;
        }
    } else {
        if (readVal != -1) {
            return true;
        }
    }

    baseVal = GetValueFromVersion(ver, "B");
    readVal = GetValueFromVersion(readVer, "B");
    if (baseVal == -1 || readVal == -1 || readVal < baseVal) {
        SHM_LOG_INFO("check driver version failed, B Build is too low:");
        return false;
    }
    return true;
}

int32_t HalGvaPrecheck(void)
{
    if (!DriverVersionCheck(DRIVER_VER_V5_END) && DriverVersionCheck(DRIVER_VER_V5_BEGIN)) {
        SHM_LOG_INFO("Driver version V5 found, compatible with V4");
        checkVer = HYBM_GVA_V4;
        return ACLSHMEM_SUCCESS;
    }
    if (DriverVersionCheck(DRIVER_VER_V4)) {
        SHM_LOG_INFO("Driver version V4 found");
        checkVer = HYBM_GVA_V4;
        return ACLSHMEM_SUCCESS;
    }
    if (DriverVersionCheck(DRIVER_VER_V3)) {
        SHM_LOG_INFO("Driver version V3 found");
        checkVer = HYBM_GVA_V3;
        return ACLSHMEM_SUCCESS;
    }
    if (DriverVersionCheck(DRIVER_VER_V2)) {
        SHM_LOG_INFO("Driver version V2 found");
        checkVer = HYBM_GVA_V2;
        return ACLSHMEM_SUCCESS;
    }
    if (DriverVersionCheck(DRIVER_VER_V1)) {
        SHM_LOG_INFO("Driver version V1 found");
        checkVer = HYBM_GVA_V1;
        return ACLSHMEM_SUCCESS;
    }

    SHM_LOG_ERROR("Failed to determine driver version");
    return ACLSHMEM_INNER_ERROR;
}
