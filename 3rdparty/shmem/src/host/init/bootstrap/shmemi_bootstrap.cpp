/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "shmemi_host_common.h"
#include "shmemi_host_def.h"
#include "dlfcn.h"
#include <arpa/inet.h>
#include <limits.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define BOOTSTRAP_MODULE_MPI "aclshmem_bootstrap_mpi.so"
#define BOOTSTRAP_MODULE_CONFIG_STORE "aclshmem_bootstrap_config_store.so"

#define BOOTSTRAP_PLUGIN_INIT_FUNC "aclshmemi_bootstrap_plugin_init"
#define BOOTSTRAP_PLUGIN_PREINIT_FUNC "aclshmemi_bootstrap_plugin_pre_init"

const char *plugin_name = nullptr;

static bool get_current_so_dir(char *dir_buf, size_t buf_len)
{
    if (dir_buf == nullptr || buf_len == 0) {
        return false;
    }

    Dl_info info;
    if (dladdr((void *)&get_current_so_dir, &info) == 0 || info.dli_fname == nullptr) {
        return false;
    }

    const char *so_path = info.dli_fname;
    const char *slash = strrchr(so_path, '/');
    if (slash == nullptr) {
        return false;
    }

    size_t dir_len = static_cast<size_t>(slash - so_path);
    if (dir_len + 1 > buf_len) {
        return false;
    }

    memcpy(dir_buf, so_path, dir_len);
    dir_buf[dir_len] = '\0';
    return true;
}

static void *safe_dlopen(const char *so_name)
{
    if (so_name == nullptr || so_name[0] == '\0') {
        SHM_LOG_ERROR("Failed to load SO: invalid so_name");
        return nullptr;
    }

    char so_dir[PATH_MAX] = {0};
    if (get_current_so_dir(so_dir, sizeof(so_dir))) {
        char full_path[PATH_MAX] = {0};
        int ret = snprintf(full_path, sizeof(full_path), "%s/%s", so_dir, so_name);
        if (ret > 0 && static_cast<size_t>(ret) < sizeof(full_path)) {
            dlerror();
            void *handle_by_full_path = dlopen(full_path, RTLD_NOW);
            if (handle_by_full_path != nullptr) {
                return handle_by_full_path;
            }
        }
    }

    dlerror();
    void *handle = dlopen(so_name, RTLD_NOW);
    const char *err = dlerror();
    if (!handle) {
        SHM_LOG_ERROR("Failed to load SO: " << so_name << ", dlerror: " << (err ? err : "unknown error"));
    }
    return handle;
}

static void safe_dlclose(void **handle)
{
    if (handle && *handle) {
        dlclose(*handle);
        *handle = nullptr;
    }
}

int bootstrap_loader_finalize(aclshmemi_bootstrap_handle_t *handle)
{
    int status = handle->finalize(handle);

    if (status != 0)
        SHM_LOG_ERROR("Bootstrap plugin finalize failed for " << plugin_name);

    dlclose(plugin_hdl);
    plugin_hdl = nullptr;

    return 0;
}

void aclshmemi_bootstrap_loader()
{
    safe_dlclose(&plugin_hdl);
    if (plugin_name) {
        plugin_hdl = safe_dlopen(plugin_name);
    }
}

void aclshmemi_bootstrap_free()
{
    safe_dlclose(&plugin_hdl);
}

int32_t aclshmemi_bootstrap_pre_init(int flags, aclshmemi_bootstrap_handle_t *handle) {
    int32_t status = ACLSHMEM_SUCCESS;
    if (flags & ACLSHMEMX_INIT_WITH_MPI) {
        SHM_LOG_ERROR("Unsupported Type for bootstrap preinit.");
        return ACLSHMEM_INVALID_PARAM;
    } else if (flags & ACLSHMEMX_INIT_WITH_UNIQUEID) {
        plugin_name = BOOTSTRAP_MODULE_CONFIG_STORE;
    } else if (flags & ACLSHMEMX_INIT_WITH_DEFAULT) {
        plugin_name = BOOTSTRAP_MODULE_CONFIG_STORE;
    } else {
        SHM_LOG_ERROR("Unknown Type for bootstrap");
        status = ACLSHMEM_INVALID_PARAM;
    }
    aclshmemi_bootstrap_loader();
    if (!plugin_hdl) {
        SHM_LOG_ERROR("Bootstrap unable to load " << plugin_name
            << ", please ensure the SO file is in the same directory as aclshmem.so.");
        aclshmemi_bootstrap_free();
        return ACLSHMEM_INVALID_VALUE;
    }
    int (*plugin_pre_init)(aclshmemi_bootstrap_handle_t *);
    dlerror();
    *((void **)&plugin_pre_init) = dlsym(plugin_hdl, BOOTSTRAP_PLUGIN_PREINIT_FUNC);
    const char *dlsym_err = dlerror();
    if (!plugin_pre_init || dlsym_err) {
        SHM_LOG_ERROR("Bootstrap plugin pre_init func dlsym failed: " << (dlsym_err ? dlsym_err : "unknown error"));
        aclshmemi_bootstrap_free();
        return ACLSHMEM_INNER_ERROR;
    }
    status = plugin_pre_init(&g_boot_handle);
    if (status != 0) {
        SHM_LOG_ERROR("Bootstrap plugin init failed for " << plugin_name);
        aclshmemi_bootstrap_free();
        return ACLSHMEM_INNER_ERROR;
    }
    return status;
}

void remove_tcp_prefix_and_copy(const char* input, char* output, size_t output_len) {
    memset(output, 0, output_len);
    if (output_len == 0) return;

    if (input == nullptr || strlen(input) == 0) {
        return;
    }

    const char* prefix_tcp = "tcp://";
    const char* prefix_tcp6 = "tcp6://";
    size_t len_tcp = strlen(prefix_tcp);
    size_t len_tcp6 = strlen(prefix_tcp6);
    const char* result_ptr = input;

    if (strncmp(input, prefix_tcp, len_tcp) == 0) {
        result_ptr = input + len_tcp;
    }
    else if (strncmp(input, prefix_tcp6, len_tcp6) == 0) {
        result_ptr = input + len_tcp6;
    }

    strncpy(output, result_ptr, output_len - 1);
    output[output_len - 1] = '\0';
}

static bool is_uid_args_valid(void *arg)
{
    if (arg == nullptr) {
        return false;
    }
    aclshmemi_bootstrap_uid_state_t *uid_args = (aclshmemi_bootstrap_uid_state_t *)arg;
    struct sockaddr *sa = reinterpret_cast<struct sockaddr *>(&uid_args->addr.addr);
    SHM_LOG_INFO("uid_args sa_family: " << sa->sa_family);
    return sa->sa_family == AF_INET || sa->sa_family == AF_INET6;
}

static bool is_valid_ip_port_url(const char *ip_port)
{
    if (ip_port == nullptr || ip_port[0] == '\0') {
        return false;
    }

    constexpr const char *tcp_prefix = "tcp://";
    constexpr const char *tcp6_prefix = "tcp6://";
    constexpr long min_port = 1024;

    std::string url(ip_port);
    std::string ip;
    std::string port_str;

    if (url.rfind(tcp_prefix, 0) == 0) {
        std::string endpoint = url.substr(strlen(tcp_prefix));
        size_t colon_pos = endpoint.rfind(':');
        if (colon_pos == std::string::npos || colon_pos == 0 || colon_pos == endpoint.size() - 1) {
            return false;
        }
        ip = endpoint.substr(0, colon_pos);
        port_str = endpoint.substr(colon_pos + 1);
    } else if (url.rfind(tcp6_prefix, 0) == 0) {
        std::string endpoint = url.substr(strlen(tcp6_prefix));
        if (endpoint.size() < 4 || endpoint[0] != '[') {
            return false;
        }
        size_t bracket_pos = endpoint.find(']');
        if (bracket_pos == std::string::npos || bracket_pos + 2 >= endpoint.size() || endpoint[bracket_pos + 1] != ':') {
            return false;
        }
        ip = endpoint.substr(1, bracket_pos - 1);
        port_str = endpoint.substr(bracket_pos + 2);
    } else {
        return false;
    }

    if (ip.empty() || port_str.empty()) {
        return false;
    }

    errno = 0;
    char *end_ptr = nullptr;
    long port = std::strtol(port_str.c_str(), &end_ptr, 10);
    if (errno == ERANGE || end_ptr == nullptr || *end_ptr != '\0' || port <= min_port || port > UINT16_MAX) {
        return false;
    }

    char addr_buf[sizeof(struct in6_addr)] = {0};
    if (inet_pton(AF_INET, ip.c_str(), addr_buf) == 1) {
        return true;
    }
    return inet_pton(AF_INET6, ip.c_str(), addr_buf) == 1;
}

int32_t aclshmemi_bootstrap_init(int flags, aclshmemx_init_attr_t *attr) {
    int32_t status = ACLSHMEM_SUCCESS;
    void *arg;
    g_boot_handle.use_attr_ipport = false;
    if (flags & ACLSHMEMX_INIT_WITH_UNIQUEID) {
        SHM_LOG_INFO("ACLSHMEMX_INIT_WITH_UNIQUEID");
        plugin_name = BOOTSTRAP_MODULE_CONFIG_STORE;
        arg = (attr != NULL) ? attr->comm_args : NULL;
        g_boot_handle.mype = attr->my_pe;
        g_boot_handle.npes = attr->n_pes;
        if (!is_uid_args_valid(arg)) {
            SHM_LOG_ERROR("BootStrap UID Mode Must Have UID !");
            return ACLSHMEM_INVALID_PARAM;
        }
    } else if (flags & ACLSHMEMX_INIT_WITH_MPI) {
        SHM_LOG_INFO("ACLSHMEMX_INIT_WITH_MPI");
        plugin_name = BOOTSTRAP_MODULE_MPI;
        arg = (attr != NULL) ? attr->comm_args : NULL;
    } else if (flags & ACLSHMEMX_INIT_WITH_DEFAULT) {
        SHM_LOG_INFO("ACLSHMEMX_INIT_WITH_DEFAULT");
        plugin_name = BOOTSTRAP_MODULE_CONFIG_STORE;
        arg = (attr != NULL) ? attr->comm_args : NULL;
        g_boot_handle.mype = attr->my_pe;
        g_boot_handle.npes = attr->n_pes;
        if (is_valid_ip_port_url(attr->ip_port)) {
            g_boot_handle.use_attr_ipport = true;
            g_boot_handle.sockFd = attr->option_attr.sockFd;
            g_boot_handle.timeOut = attr->option_attr.shm_init_timeout;
            g_boot_handle.timeControlOut = attr->option_attr.control_operation_timeout;
            strncpy(g_boot_handle.ipport, attr->ip_port, sizeof(g_boot_handle.ipport) - 1);
            g_boot_handle.ipport[sizeof(g_boot_handle.ipport) - 1] = '\0';
            SHM_LOG_INFO("Default bootstrap will use attr ip_port directly: " << g_boot_handle.ipport);
        } else if (is_uid_args_valid(arg)) {
            SHM_LOG_INFO("BootStrap Default Mode got invalid ip_port: "
                << (attr->ip_port[0] != '\0' ? attr->ip_port : "<empty>")
                << ", fallback to UNIQUEID bootstrap args.");
            g_boot_handle.use_attr_ipport = false;
        } else {
            SHM_LOG_ERROR("BootStrap Default Mode got invalid ip_port: "
                << (attr->ip_port[0] != '\0' ? attr->ip_port : "<empty>")
                << ", and no valid UNIQUEID comm_args fallback.");
            return ACLSHMEM_INVALID_PARAM;
        }
    } else {
        SHM_LOG_ERROR("Unknown Type for bootstrap");
        return ACLSHMEM_INVALID_PARAM;
    }
    aclshmemi_bootstrap_loader();
    if (!plugin_hdl) {
        SHM_LOG_ERROR("Bootstrap unable to load " << plugin_name
            << ", please ensure the SO file is in the same directory as aclshmem.so.");
        aclshmemi_bootstrap_free();
        return ACLSHMEM_INVALID_VALUE;
    }
    int (*plugin_init)(void *, aclshmemi_bootstrap_handle_t *);
    dlerror();
    *((void **)&plugin_init) = dlsym(plugin_hdl, BOOTSTRAP_PLUGIN_INIT_FUNC);
    const char *dlsym_err = dlerror();
    if (!plugin_init || dlsym_err) {
        SHM_LOG_ERROR("Bootstrap plugin init func dlsym failed: " << (dlsym_err ? dlsym_err : "unknown error"));
        aclshmemi_bootstrap_free();
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_INFO("Calling plugin_init for " << plugin_name
        << ", use_attr_ipport=" << g_boot_handle.use_attr_ipport
        << ", mype=" << g_boot_handle.mype
        << ", npes=" << g_boot_handle.npes);
    status = plugin_init(arg, &g_boot_handle);
    SHM_LOG_INFO("plugin_init returned status=" << status << " for " << plugin_name);
    if (status != 0) {
        SHM_LOG_ERROR("Bootstrap plugin init failed for " << plugin_name);
        aclshmemi_bootstrap_free();
        return ACLSHMEM_INNER_ERROR;
    }
    g_boot_handle.is_bootstraped = true;
    return status;
}

void aclshmemi_bootstrap_finalize() {
    if (!g_boot_handle.is_bootstraped) {
        return;
    }
    g_boot_handle.finalize(&g_boot_handle);
    g_boot_handle.is_bootstraped = false;
    safe_dlclose(&plugin_hdl);
}