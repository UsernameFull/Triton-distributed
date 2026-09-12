/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_SHM_FILE_UTIL_H
#define SHMEM_SHM_FILE_UTIL_H

#include <cstring>
#include <dirent.h>
#include <string>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iomanip>
#include "shmemi_logger.h"

#define PATH_MAX_LIMIT 4096L
namespace shm {
namespace utils {
class FileUtil {
    static constexpr uint32_t MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
public:
    /**
       * @brief Get the lengthiest of path
       *
       * @return the lengthiest of path
     */
    static constexpr size_t GetSafePathMax();

    /**
     * @brief Check if file or dir exists
     */
    static bool Exist(const std::string &path);

    /**
     * @brief Check if the file or dir readable
     */
    static bool Readable(const std::string &path);

    /**
     * @brief Check if the file or dir writable
     */
    static bool Writable(const std::string &path);

    /**
     * @brief Create dir
     */
    static bool MakeDir(const std::string &path, uint32_t mode);

    /**
     * @brief Remove the dir without sub dirs
     */
    static bool Remove(const std::string &path, bool canonicalPath = true);

    /**
     * @brief Remove the dir recursively, its sub dir will be removed
     */
    static bool RemoveDirRecursive(const std::string &path);

    /**
     * @brief Get the realpath for security consideration
     */
    static bool Realpath(std::string &path);

    /**
     * @brief Get real path of a library and check if exists
     *
     * @param libDirPath   [in] dir path of the library
     * @param libName      [in] library name
     * @param realPath     [out] realpath of the library
     * @return 0 if successful
     */
    static bool LibraryRealPath(const std::string &libDirPath, const std::string &libName, std::string &realPath);

    /**
     * @brief Get size of a file
     */
    static size_t GetFileSize(const std::string &filePath);

    /**
     * @brief Close file
     */
    static void CloseFile(FILE* fp);

    /**
     * @brief Check if the file or dir is symbol link
     */
    static bool IsSymlink(const std::string &filePath);

    /**
     * @brief Check if the file is empty one
     */
    static bool IsEmptyFile(const std::string &filePath);

    /**
     * @brief Find whether the path is a file or not
     *
     * @param path         [in] input path
     * @return true if it is a file
     */
    static bool IsFile(const std::string &path);

    /**
     * @brief Find whether the path is a directory or not
     *
     * @param path         [in] input path
     * @return true if it is a directory
     */
    static bool IsDir(const std::string &path);

    /**
     * @brief Check if the path is owned by current user or root
     *        Security check to prevent loading libraries from untrusted paths
     *
     * @param path         [in] path to check
     * @return true if owned by current user or root
     */
    static bool IsOwnedByCurrentUserOrRoot(const std::string &path);

    /**
     * @brief Check if the path has secure permissions (no write permission for group/others)
     *        Security check to prevent tampering with libraries
     *
     * @param path         [in] path to check
     * @return true if permissions are secure
     */
    static bool HasSecurePermissions(const std::string &path);

    /**
     * @brief Find whether the path exceed the max size or not
     *
     * @param path         [in] input path
     * @param maxSize      [in] the max size allowed
     * @return true if the file size is less or equals to maxSize
     */
    static bool CheckFileSize(const std::string &path, uint32_t maxSize = MAX_FILE_SIZE);
};

inline bool FileUtil::Exist(const std::string &path)
{
    return access(path.c_str(), 0) != -1;
}

inline bool FileUtil::Readable(const std::string &path)
{
    return access(path.c_str(), F_OK | R_OK) != -1;
}

inline bool FileUtil::Writable(const std::string &path)
{
    return access(path.c_str(), F_OK | W_OK) != -1;
}

inline bool FileUtil::MakeDir(const std::string &path, uint32_t mode)
{
    if (path.empty()) {
        return false;
    }

    if (Exist(path)) {
        return true;
    }

    return ::mkdir(path.c_str(), mode) == 0;
}

inline bool FileUtil::Remove(const std::string &path, bool canonicalPath)
{
    if (path.empty() || path.size() > PATH_MAX_LIMIT) {
        return false;
    }

    std::string realPath = path;
    if (canonicalPath && !Realpath(realPath)) {
        return false;
    }

    return ::remove(realPath.c_str()) == 0;
}

inline bool FileUtil::Realpath(std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX_LIMIT) {
        return false;
    }

    /* It will allocate memory to store path */
    char* tmp = new char[shm::utils::FileUtil::GetSafePathMax() + 1];
    char* realPath = realpath(path.c_str(), tmp);
    if (realPath == nullptr) {
        delete[] tmp;
        return false;
    }

    path = realPath;
    realPath = nullptr;
    delete[] tmp;
    return true;
}

inline bool FileUtil::LibraryRealPath(const std::string &libDirPath, const std::string &libName, std::string &realPath)
{
    std::string tmpFullPath = libDirPath;
    if (!Realpath(tmpFullPath)) {
        return false;
    }

    if (tmpFullPath.back() != '/') {
        tmpFullPath.push_back('/');
    }

    tmpFullPath.append(libName);
    auto ret = ::access(tmpFullPath.c_str(), F_OK);
    if (ret != 0) {
        return false;
    }

    realPath = tmpFullPath;
    return true;
}

inline void FileUtil::CloseFile(FILE* fp)
{
    if (fp == nullptr) {
        return;
    }

    auto ret = fclose(fp);
    if (ret != 0) {
        SHM_LOG_INFO("util fclose failed, ret = " << ret);
    }
}

inline size_t FileUtil::GetFileSize(const std::string &path)
{
    if (!Exist(path)) {
        return 0;
    }

    std::string realFilePath = path;
    if (!Realpath(realFilePath)) {
        return 0;
    }

    FILE* fp = fopen(realFilePath.c_str(), "rb");
    if (fp == nullptr) {
        return 0;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        CloseFile(fp);
        return 0;
    }

    long fileSizeLong = ftell(fp);
    if (fileSizeLong == -1L) {
        SHM_LOG_ERROR("ftell() failed for file: " << realFilePath);
        CloseFile(fp);
        return 0;
    }
    size_t fileSize = static_cast<size_t>(fileSizeLong);
    if (fseek(fp, 0, SEEK_END) != 0) {
        CloseFile(fp);
        return 0;
    }

    CloseFile(fp);

    return fileSize;
}

inline bool FileUtil::IsSymlink(const std::string &filePath)
{
    /* remove / at tail */
    std::string cleanPath = filePath;
    while (!cleanPath.empty() && cleanPath.back() == '/') {
        cleanPath.pop_back();
    }

    struct stat buf;
    if (lstat(cleanPath.c_str(), &buf) != 0) {
        return false;
    }
    return S_ISLNK(buf.st_mode);
}

inline bool FileUtil::IsEmptyFile(const std::string &filePath)
{
    if (!Exist(filePath)) {
        return false;
    }

    return GetFileSize(filePath) == 0;
}

inline bool FileUtil::IsFile(const std::string &path)
{
    struct stat buf;
    if (lstat(path.c_str(), &buf) != 0) {
        return false;
    }
    return S_ISREG(buf.st_mode);
}

inline bool FileUtil::IsDir(const std::string &path)
{
    struct stat buf;
    if (lstat(path.c_str(), &buf) != 0) {
        return false;
    }
    return S_ISDIR(buf.st_mode);
}

inline bool FileUtil::IsOwnedByCurrentUserOrRoot(const std::string &path)
{
    struct stat buf;
    if (stat(path.c_str(), &buf) != 0) {
        SHM_LOG_ERROR("stat failed for path: " << path);
        return false;
    }
    
    uid_t currentUid = getuid();
    uid_t pathOwner = buf.st_uid;
    
    if (pathOwner == currentUid || pathOwner == 0) {
        return true;
    }
    
    SHM_LOG_ERROR("Path owner check failed: path=" << path 
                  << ", owner_uid=" << pathOwner 
                  << ", current_uid=" << currentUid);
    return false;
}

inline bool FileUtil::HasSecurePermissions(const std::string &path)
{
    struct stat buf;
    if (stat(path.c_str(), &buf) != 0) {
        SHM_LOG_ERROR("stat failed for path: " << path);
        return false;
    }
    
    mode_t mode = buf.st_mode;
    
    if ((mode & S_IWGRP) || (mode & S_IWOTH)) {
        SHM_LOG_ERROR("Insecure permissions: path=" << path 
                      << ", mode=" << std::oct << mode 
                      << " (group or others have write permission)");
        return false;
    }
    
    return true;
}

inline bool FileUtil::CheckFileSize(const std::string &path, uint32_t maxSize)
{
    if (!Exist(path)) {
        return false;
    }

    return GetFileSize(path) <= static_cast<size_t>(maxSize);
}

inline constexpr size_t FileUtil::GetSafePathMax()
{
#ifdef PATH_MAX
    return (PATH_MAX < PATH_MAX_LIMIT) ? PATH_MAX : PATH_MAX_LIMIT;
#else
    return PATH_MAX_LIMIT;
#endif
}
}  // namespace utils
}  // namespace shm

#endif