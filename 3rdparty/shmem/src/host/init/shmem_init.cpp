/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <sstream>
#include <functional>
#include <algorithm>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "shmemi_host_common.h"
#include "shmemi_init.h"
#include "host/shmem_host_def.h"
#include "prof/prof_util.h"

using namespace std;

#define DEFAULT_MY_PE (-1)
#define DEFAULT_N_PES (-1)

constexpr int DEFAULT_TEVENT = 0;
constexpr int DEFAULT_BLOCK_NUM = 1;

constexpr uint32_t DEFAULT_MTE_UB_SIZE = 16 * 1024;
constexpr uint32_t DEFAULT_SDMA_UB_SIZE = 64;
constexpr int64_t DEFAULT_SDMA_UB_OFFSET = 191 * 1024;
constexpr uint32_t DEFAULT_RDMA_UB_SIZE = 64;
constexpr int64_t DEFAULT_RDMA_UB_OFFSET = 190 * 1024;

// initializer
#define ACLSHMEM_DEVICE_HOST_STATE_INITIALIZER                                                    \
    {                                                                                             \
        (1 << 16) + sizeof(aclshmem_device_host_state_t),  /* version */                          \
        (DEFAULT_MY_PE),                                   /* mype */                             \
        (DEFAULT_N_PES),                                   /* npes */                             \
        NULL,                                              /* heap_base */                        \
        NULL,                                              /* host_heap_base */                   \
        NULL,                                              /* p2p_device_heap_base */             \
        NULL,                                              /* rdma_device_heap_base */            \
        NULL,                                              /* sdma_device_heap_base */            \
        NULL,                                              /* p2p_heap_host_base */               \
        NULL,                                              /* rdma_heap_host_base */              \
        NULL,                                              /* sdma_heap_host_base */              \
        {},                                                /* topo_list */                        \
        SIZE_MAX,                                          /* heap_size */                        \
        {NULL},                                            /* team_pools */                       \
        0,                                                 /* sync_pool */                        \
        0,                                                 /* sync_counter */                     \
        0,                                                 /* core_sync_pool */                   \
        0,                                                 /* core_sync_counter */                \
        false,                                             /* aclshmem_is_aclshmem_initialized */ \
        false,                                             /* aclshmem_is_aclshmem_created */     \
        {0, DEFAULT_MTE_UB_SIZE, 0},                       /* aclshmem_mte_config */              \
        {DEFAULT_SDMA_UB_OFFSET, DEFAULT_SDMA_UB_SIZE, 0}, /* aclshmem_sdma_config */             \
        {DEFAULT_RDMA_UB_OFFSET, DEFAULT_RDMA_UB_SIZE, 0}, /* aclshmem_rdma_config */             \
        0,                                                 /* qp_info */                          \
        0,                                                 /* sdma_workspace_addr */              \
        NULL,                                              /* aclshmem_prof_pe_t */               \
        0,                                                 /* signal_addr */                      \
    }

aclshmem_device_host_state_t g_state = ACLSHMEM_DEVICE_HOST_STATE_INITIALIZER;
aclshmem_host_state_t g_state_host = {nullptr, DEFAULT_TEVENT, DEFAULT_BLOCK_NUM, 0};
aclshmem_prof_pe_t g_host_profs;

// bootstrap plugin_hdl
void* plugin_hdl = nullptr;
aclshmemi_bootstrap_handle_t g_boot_handle;
std::shared_ptr<memory_manager> aclshmemi_memory_manager = nullptr;
std::shared_ptr<memory_manager> aclshmemi_host_memory_manager = nullptr;

// Count Used to guard Multi instance scenario
static int g_init_manager_count = 0;
aclshmemi_init_backend* init_manager = nullptr;

// Protect instance context access
static std::mutex g_aclshmem_ctx_mutex;

// Instance context used to store global_resources
struct aclshmem_context {
    uint64_t instance_id = 0;

    aclshmem_device_host_state_t state = ACLSHMEM_DEVICE_HOST_STATE_INITIALIZER;
    aclshmem_host_state_t host_state = {nullptr, DEFAULT_TEVENT, DEFAULT_BLOCK_NUM};

    void* bootstrap_plugin_hdl = nullptr;
    aclshmemi_bootstrap_handle_t boot_handle;

    std::shared_ptr<memory_manager> dev_mem_manager = nullptr;
    std::shared_ptr<memory_manager> host_mem_manager = nullptr;

    explicit aclshmem_context(uint64_t id) : instance_id(id) {}
};

// g_instance_ctx for global use
aclshmem_instance_ctx* g_instance_ctx = new aclshmem_instance_ctx{0, nullptr};

std::map<uint32_t, aclshmem_instance_ctx*> aclshmem_ctx_domain = {{0, g_instance_ctx}};
std::map<uint32_t, aclshmem_context*> aclshmem_resource_domain = {{0, new aclshmem_context(0)}};

namespace {
bool aclshmemi_parse_port_strict(const std::string& port_str, uint16_t& port)
{
    constexpr int max_port = 65535;
    try {
        size_t parsed_len = 0;
        int port_int = std::stoi(port_str, &parsed_len);
        if (parsed_len != port_str.size() || port_int < 0 || port_int > max_port) {
            return false;
        }
        port = static_cast<uint16_t>(port_int);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}
}

int32_t version_compatible()
{
    int32_t status = ACLSHMEM_SUCCESS;
    return status;
}

int32_t aclshmemi_state_init_attr(aclshmemx_init_attr_t* attributes)
{
    int32_t status = ACLSHMEM_SUCCESS;
    g_state.mype = attributes->my_pe;
    g_state.npes = attributes->n_pes;
    g_state.heap_size = attributes->local_mem_size + ACLSHMEM_EXTRA_SIZE;

    aclrtStream stream = nullptr;
    ACLSHMEM_CHECK_RET(aclrtCreateStream(&stream));
    g_state_host.default_stream = stream;
    g_state_host.default_event_id = DEFAULT_TEVENT;
    g_state_host.default_block_num = DEFAULT_BLOCK_NUM;
    return status;
}

bool is_valid_data_op_engine_type(data_op_engine_type_t value)
{
    constexpr uint32_t valid_mask = (static_cast<uint32_t>(ACLSHMEM_DATA_OP_MAX) << 1) - 1;
    uint32_t int_value = static_cast<uint32_t>(value);
    return int_value > 0 && (int_value & ~valid_mask) == 0;
}

int32_t check_attr(aclshmemx_init_attr_t* attributes)
{
    SHM_LOG_DEBUG(
        "check_attr my_pe=" << attributes->my_pe << " n_pes=" << attributes->n_pes
                            << " local_mem_size=" << attributes->local_mem_size
                            << " shm_init_timeout=" << attributes->option_attr.shm_init_timeout
                            << " control_operation_timeout=" << attributes->option_attr.control_operation_timeout);
    SHM_VALIDATE_RETURN(attributes->my_pe >= 0, "my_pe is less than zero", ACLSHMEM_INVALID_VALUE);
    SHM_VALIDATE_RETURN(attributes->n_pes > 0, "n_pes is less than or equal to zero", ACLSHMEM_INVALID_VALUE);
    SHM_VALIDATE_RETURN(attributes->n_pes <= ACLSHMEM_MAX_PES, "n_pes is too large", ACLSHMEM_INVALID_VALUE);
    SHM_VALIDATE_RETURN(
        attributes->my_pe < attributes->n_pes, "my_pe is greater than or equal to n_pes", ACLSHMEM_INVALID_PARAM);
    SHM_VALIDATE_RETURN(
        attributes->local_mem_size > 0, "local_mem_size less than or equal to 0", ACLSHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(attributes->local_mem_size <= ACLSHMEM_MAX_LOCAL_SIZE, ACLSHMEM_INVALID_VALUE);

    SHM_VALIDATE_RETURN(
        attributes->option_attr.shm_init_timeout != 0, "shm_init_timeout is zero", ACLSHMEM_INVALID_VALUE);
    SHM_VALIDATE_RETURN(
        attributes->option_attr.control_operation_timeout != 0, "control_operation_timeout is zero",
        ACLSHMEM_INVALID_VALUE);
    SHM_VALIDATE_RETURN(attributes->option_attr.data_op_engine_type > 0, "sockFd is invalid", ACLSHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(
        is_valid_data_op_engine_type(attributes->option_attr.data_op_engine_type), ACLSHMEM_INVALID_VALUE);
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_control_barrier_all() { return init_manager->aclshmemi_control_barrier_all(); }

int32_t is_alloc_size_symmetric(size_t size) { return init_manager->is_alloc_size_symmetric(size); }

int32_t update_device_state()
{
    return init_manager->update_device_state((void*)&g_state, sizeof(aclshmem_device_host_state_t));
}

int32_t aclshmemx_init_status(void)
{
    if (!g_state.is_aclshmem_created)
        return ACLSHMEM_STATUS_NOT_INITIALIZED;
    else if (!g_state.is_aclshmem_initialized)
        return ACLSHMEM_STATUS_SHM_CREATED;
    else if (g_state.is_aclshmem_initialized)
        return ACLSHMEM_STATUS_IS_INITIALIZED;
    else
        return ACLSHMEM_STATUS_INVALID;
}

int aclshmemx_set_attr_uniqueid_args(
    int my_pe, int n_pes, int64_t local_mem_size, aclshmemx_uniqueid_t* uid, aclshmemx_init_attr_t* aclshmem_attr)
{
    /* Save to uid_args */
    SHM_ASSERT_RETURN(local_mem_size <= ACLSHMEM_MAX_LOCAL_SIZE, ACLSHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(n_pes <= ACLSHMEM_MAX_PES, ACLSHMEM_INVALID_VALUE);
    SHM_ASSERT_RETURN(my_pe < ACLSHMEM_MAX_PES, ACLSHMEM_INVALID_VALUE);
    aclshmemi_bootstrap_uid_state_t* uid_args = (aclshmemi_bootstrap_uid_state_t*)(uid);
    void* comm_args = reinterpret_cast<void*>(uid_args);
    aclshmem_attr->comm_args = comm_args;
    aclshmem_attr->my_pe = my_pe;
    aclshmem_attr->n_pes = n_pes;
    aclshmem_attr->local_mem_size = local_mem_size;

    return ACLSHMEM_SUCCESS;
}

bool check_support_d2h()
{
    int32_t major_version = -1;
    int32_t minor_version = -1;
    int32_t patch_version = -1;
    if (aclrtGetVersion(&major_version, &minor_version, &patch_version) != 0) {
        SHM_LOG_INFO("aclrtGetVersion failed, disable d2h feature");
        return false;
    }
    if (major_version <= 1 && minor_version < 15) {
        SHM_LOG_INFO(
            "The current AscendCL version is " << major_version << "." << minor_version << "." << patch_version
                                               << ", which does not support d2h.");
        return false;
    }
    return true;
}

int32_t aclshmemi_signal_finalize()
{
    if (g_state.signal_addr != 0) {
        aclshmem_free(reinterpret_cast<void*>(g_state.signal_addr));
        g_state.signal_addr = 0;
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_signal_init()
{
    constexpr size_t align_size = 512;
    g_state.signal_addr = (uint64_t)aclshmem_malloc(align_size);
    if (g_state.signal_addr == 0) {
        SHM_LOG_ERROR("malloc signal failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    auto ret = aclrtMemset((void*)g_state.signal_addr, ACLSHMEM_SIGNAL_SIZE, 0, ACLSHMEM_SIGNAL_SIZE);
    if (ret != 0) {
        aclshmemi_signal_finalize();
        SHM_LOG_ERROR("memset signal failed. ret=" << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

static int32_t aclshmemi_instance_port_selection(aclshmemx_init_attr_t* attributes)
{
    // 1. Get Port Range From Environment
    const char* env_port_range = std::getenv("SHMEM_INSTANCE_PORT_RANGE");
    if (env_port_range == nullptr) {
        SHM_LOG_ERROR("The environment variable SHMEM_INSTANCE_PORT_RANGE is not set.");
        return ACLSHMEM_INVALID_VALUE;
    }

    std::string env_port_range_str(env_port_range);
    std::size_t env_pos = env_port_range_str.find(':');
    if (env_pos == std::string::npos) {
        SHM_LOG_ERROR("SHMEM_INSTANCE_PORT_RANGE format should be start_port:end_port");
        return ACLSHMEM_INVALID_VALUE;
    }

    uint16_t start_port;
    uint16_t end_port;
    std::string start_port_str = env_port_range_str.substr(0, env_pos);
    std::string end_port_str = env_port_range_str.substr(env_pos + 1, env_port_range_str.size());
    if (!aclshmemi_parse_port_strict(start_port_str, start_port)) {
        SHM_LOG_ERROR("Invalid start_port value: " << start_port_str << ", must be in range [0, 65535]");
        return ACLSHMEM_INVALID_VALUE;
    }
    if (!aclshmemi_parse_port_strict(end_port_str, end_port)) {
        SHM_LOG_ERROR("Invalid end_port value: " << end_port_str << ", must be in range [0, 65535]");
        return ACLSHMEM_INVALID_VALUE;
    }
    if (end_port < start_port) {
        SHM_LOG_ERROR("SHMEM_INSTANCE_PORT_RANGE: start_port " << start_port << " exceeds end_port " << end_port);
        return ACLSHMEM_INVALID_VALUE;
    }

    // 2. Check if instance_id is valid
    int max_instance_num = end_port - start_port;
    uint64_t instance_id = attributes->instance_id;
    if (instance_id > static_cast<uint64_t>(max_instance_num)) {
        SHM_LOG_ERROR(
            "instance_id " << instance_id << " exceeds max_instance_num " << max_instance_num << " in default mode");
        return ACLSHMEM_INVALID_VALUE;
    }
    uint16_t port = start_port + static_cast<uint16_t>(instance_id);

    // 3. If ip_port is null, return
    if (attributes->ip_port[0] == '\0') {
        SHM_LOG_ERROR("init with my_rank:" << attributes->my_pe << " ip_port is nullptr!");
        return ACLSHMEM_INVALID_VALUE;
    }

    // 4. replace ip_port in attributes
    std::string ip_port_str(attributes->ip_port);
    std::size_t pos = ip_port_str.find(':', ip_port_str.find(':') + 1);
    if (pos == std::string::npos) {
        SHM_LOG_ERROR("ip_port format should be ip:port");
        return ACLSHMEM_INVALID_VALUE;
    }

    uint16_t input_port;
    std::string input_port_str = ip_port_str.substr(pos + 1, ip_port_str.size());
    if (!aclshmemi_parse_port_strict(input_port_str, input_port)) {
        SHM_LOG_ERROR("Invalid input_port value: " << input_port_str << ", must be in range [0, 65535]");
        return ACLSHMEM_INVALID_VALUE;
    }
    if (input_port != 0) {
        SHM_LOG_ERROR("input_port must be 0 in default mode, but input_port is " << input_port);
        return ACLSHMEM_INVALID_VALUE;
    }

    std::string instance_ipport = ip_port_str.substr(0, pos) + ":" + std::to_string(port);
    if (instance_ipport.empty()) {
        SHM_LOG_ERROR("my_rank:" << attributes->my_pe << " instance ipport is nullptr!");
        return ACLSHMEM_INVALID_VALUE;
    }

    // 5. Write port in attributes
    size_t ipport_len = std::min(instance_ipport.size(), static_cast<std::size_t>(ACLSHMEM_MAX_IP_PORT_LEN - 1));
    std::copy_n(instance_ipport.c_str(), ipport_len, attributes->ip_port);
    attributes->ip_port[ipport_len] = '\0';

    return ACLSHMEM_SUCCESS;
}

static int aclshmemi_instance_ctx_create(aclshmemx_init_attr_t* attributes)
{
    uint64_t instance_id = attributes->instance_id;
    if (instance_id == 0) {
        // Do Nothing.
        return ACLSHMEM_SUCCESS;
    }

    // Check if Repeat
    if (aclshmem_ctx_domain.find(instance_id) != aclshmem_ctx_domain.end() ||
        aclshmem_resource_domain.find(instance_id) != aclshmem_resource_domain.end()) {
        SHM_LOG_WARN("Instance " << instance_id << " already exists ! Please don't repeat create !");
        return ACLSHMEM_SUCCESS;
    }

    // aclshmem_instance port selection, if init with uid, skip
    if (attributes->ip_port[0] != '\0' || attributes->comm_args == nullptr) {
        ACLSHMEM_CHECK_RET(aclshmemi_instance_port_selection(attributes));
    }

    // Create aclshmem_instance_ctx
    aclshmem_instance_ctx* aclshmem_ctx = new aclshmem_instance_ctx{attributes->instance_id, nullptr};
    if (aclshmem_ctx == nullptr) {
        SHM_LOG_ERROR("Failed to allocate memory for aclshmem_instance_ctx.");
        return ACLSHMEM_INNER_ERROR;
    }
    aclshmem_ctx_domain[attributes->instance_id] = aclshmem_ctx;

    aclshmem_context* aclshmem_resource_ctx = new aclshmem_context(attributes->instance_id);
    if (aclshmem_resource_ctx == nullptr) {
        SHM_LOG_ERROR("Failed to allocate memory for aclshmem_context.");
        return ACLSHMEM_INNER_ERROR;
    }
    aclshmem_resource_domain[attributes->instance_id] = aclshmem_resource_ctx;

    SHM_LOG_WARN(
        "PE: " << attributes->my_pe << " USING SHMEM Multi-Instance Mode ! Now Ctx set to Instance " << instance_id
               << " !");
    return ACLSHMEM_SUCCESS;
}

static int aclshmemi_instance_ctx_destroy(uint64_t instance_id)
{
    if (instance_id == 0) {
        // Do Nothing.
        return ACLSHMEM_SUCCESS;
    }

    // if destroy an active context
    if (instance_id == g_instance_ctx->id) {
        // Set active context to Instance 0
        aclshmemx_instance_ctx_set_impl(0);
    }

    // Release Resource
    auto it_resource = aclshmem_resource_domain.find(instance_id);
    if (it_resource == aclshmem_resource_domain.end()) {
        SHM_LOG_ERROR("Instance " << instance_id << " not exists! Illegal Context Destroy!");
        return ACLSHMEM_INNER_ERROR;
    }
    delete it_resource->second;
    aclshmem_resource_domain.erase(it_resource);

    auto it_ctx = aclshmem_ctx_domain.find(instance_id);
    if (it_ctx == aclshmem_ctx_domain.end()) {
        SHM_LOG_ERROR("Instance " << instance_id << " not exists! Illegal Context Destroy!");
        return ACLSHMEM_INNER_ERROR;
    }
    delete it_ctx->second;
    aclshmem_ctx_domain.erase(it_ctx);

    return ACLSHMEM_SUCCESS;
}

aclshmem_instance_ctx* aclshmemx_instance_ctx_get()
{
    std::lock_guard<std::mutex> lock(g_aclshmem_ctx_mutex);
    return g_instance_ctx;
}

int aclshmemx_instance_ctx_set(uint64_t instance_id)
{
    std::lock_guard<std::mutex> lock(g_aclshmem_ctx_mutex);
    return aclshmemx_instance_ctx_set_impl(instance_id);
}

// Inner Context set Interface, No Lock.
int aclshmemx_instance_ctx_set_impl(uint64_t instance_id)
{
    if (aclshmem_ctx_domain.find(instance_id) != aclshmem_ctx_domain.end()) {
        aclshmem_instance_ctx* new_ctx = aclshmem_ctx_domain[instance_id];
        aclshmem_context* new_context = aclshmem_resource_domain[instance_id];
        aclshmem_context* current_context = aclshmem_resource_domain[g_instance_ctx->id];

        // Global_vars write back
        current_context->state = g_state;
        current_context->host_state = g_state_host;
        current_context->bootstrap_plugin_hdl = plugin_hdl;
        current_context->boot_handle = g_boot_handle;
        current_context->dev_mem_manager = aclshmemi_memory_manager;
        current_context->host_mem_manager = aclshmemi_host_memory_manager;

        // Set Global_vars by new_context
        g_state = new_context->state;
        g_state_host = new_context->host_state;
        plugin_hdl = new_context->bootstrap_plugin_hdl;
        g_boot_handle = new_context->boot_handle;
        aclshmemi_memory_manager = new_context->dev_mem_manager;
        aclshmemi_host_memory_manager = new_context->host_mem_manager;

        // Set New aclshmem context
        g_instance_ctx = aclshmem_ctx_domain[instance_id];

        return ACLSHMEM_SUCCESS;
    }
    SHM_LOG_ERROR("Context Set failed ! Can't find instance " << instance_id << " !");
    return ACLSHMEM_INNER_ERROR;
}

int32_t aclshmemx_init_attr(aclshmemx_bootstrap_t bootstrap_flags, aclshmemx_init_attr_t* attributes)
{
    std::lock_guard<std::mutex> lock(g_aclshmem_ctx_mutex);
    int32_t ret;
    uint64_t id = attributes->instance_id;

    // Multi-instance create ctx
    ACLSHMEM_CHECK_RET(aclshmemi_instance_ctx_create(attributes));
    ACLSHMEM_CHECK_RET(aclshmemx_instance_ctx_set_impl(id));

    // config init
    SHM_ASSERT_RETURN(attributes != nullptr, ACLSHMEM_INVALID_PARAM);
    ACLSHMEM_CHECK_RET(
        aclshmemx_init_status() != ACLSHMEM_STATUS_NOT_INITIALIZED,
        "SHMEM has been initialized, do not call init interface repeatedly!", ACLSHMEM_INNER_ERROR);
    ACLSHMEM_CHECK_RET(aclshmemx_set_log_level(aclshmem_log::ERROR_LEVEL));
    ACLSHMEM_CHECK_RET(
        check_attr(attributes),
        "An error occurred while checking the initialization attributes. Please check the initialization parameters.");
    ACLSHMEM_CHECK_RET(version_compatible(), "ACLSHMEM Version mismatch.");
    // init bootstrap
    ACLSHMEM_CHECK_RET(aclshmemi_bootstrap_init(bootstrap_flags, attributes));
    ACLSHMEM_CHECK_RET(aclshmemi_state_init_attr(attributes));

    // init backend for memory manager
    g_init_manager_count++;
    if (init_manager == nullptr) {
        init_manager = new aclshmemi_init_backend();
    } else {
        SHM_LOG_INFO("init_manager already exists, skipping creation. ");
    }

    // aclshmem_entity init
    ACLSHMEM_CHECK_RET(init_manager->bind_aclshmem_entity(attributes, &g_state, &g_boot_handle));
    ACLSHMEM_CHECK_RET(init_manager->init_device_state());
    ACLSHMEM_CHECK_RET(init_manager->reserve_heap());
    ACLSHMEM_CHECK_RET(init_manager->setup_heap());

    // shmem submodules init
    ACLSHMEM_CHECK_RET(memory_manager_initialize(g_state.heap_base, g_state.heap_size));

#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE
    if (check_support_d2h()) {
        // only reserve dramp heap, skip setup_heap for host dram, setup heap when malloc on host
        ACLSHMEM_CHECK_RET(init_manager->reserve_heap(HOST_SIDE));
    }
#endif
    ACLSHMEM_CHECK_RET(aclshmemi_signal_init());
    ACLSHMEM_CHECK_RET(aclshmemi_team_init(g_state.mype, g_state.npes));
    ACLSHMEM_CHECK_RET(aclshmemi_sync_init());
    g_state.is_aclshmem_initialized = true;
    ACLSHMEM_CHECK_RET(prof_util_init(&g_host_profs, &g_state));
    ACLSHMEM_CHECK_RET(update_device_state());
    ACLSHMEM_CHECK_RET(aclshmemi_control_barrier_all());
    SHM_LOG_INFO("The ACLSHMEM pe: " << aclshmem_my_pe() << " init success.");
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmem_finalize(uint64_t instance_id)
{
    std::lock_guard<std::mutex> lock(g_aclshmem_ctx_mutex);
    // When aclshmem_finalize, first set context; otherwise we will finalize unknown instance
    aclshmemx_instance_ctx_set_impl(instance_id);

    SHM_LOG_INFO("The instance : " << instance_id << ", The pe: " << aclshmem_my_pe() << " begins to finalize.");
    if (init_manager == nullptr) {
        SHM_LOG_INFO("init_manager is null finalize success.");
        g_state.is_aclshmem_initialized = false;
        ACLSHMEM_CHECK_RET(aclshmemi_instance_ctx_destroy(instance_id));
        return ACLSHMEM_SUCCESS;
    }
    // shmem submodules finalize
    ACLSHMEM_CHECK_RET(aclshmemi_team_finalize());
    ACLSHMEM_CHECK_RET(aclshmemi_signal_finalize());
    memory_manager_destroy();
    // shmem basic finalize
    ACLSHMEM_CHECK_RET(init_manager->remove_heap());
    ACLSHMEM_CHECK_RET(init_manager->release_heap());
    ACLSHMEM_CHECK_RET(init_manager->finalize_device_state());
    SHM_LOG_INFO("release_heap success.");
#ifdef HAS_ACLRT_MEM_FABRIC_HANDLE
    if (check_support_d2h()) {
        ACLSHMEM_CHECK_RET(init_manager->remove_heap(HOST_SIDE));
        ACLSHMEM_CHECK_RET(init_manager->release_heap(HOST_SIDE));
    }
#endif
    ACLSHMEM_CHECK_RET(init_manager->release_aclshmem_entity(instance_id));
    if (g_state_host.default_stream != nullptr) {
        auto ret = aclrtSynchronizeStream(g_state_host.default_stream);
        if (ret != 0) {
            SHM_LOG_ERROR("Synchronize stream failed. ret=" << ret);
        }
        ret = aclrtDestroyStream(g_state_host.default_stream);
        if (ret != 0) {
            SHM_LOG_ERROR("Destroy stream failed. ret=" << ret);
        }
        g_state_host.default_stream = nullptr;
    }
    aclshmemi_bootstrap_finalize();

    // Only When Process have no instance, then release init_manager.
    g_init_manager_count--;
    if (g_init_manager_count == 0) {
        delete init_manager;
        init_manager = nullptr;
    }
    SHM_LOG_INFO("The pe: " << aclshmem_my_pe() << " finalize success.");
    g_state.is_aclshmem_initialized = false;
    ACLSHMEM_CHECK_RET(aclshmemi_instance_ctx_destroy(instance_id));
    return ACLSHMEM_SUCCESS;
}

void aclshmem_info_get_version(int* major, int* minor)
{
    SHM_ASSERT_RET_VOID(major != nullptr && minor != nullptr);
    *major = ACLSHMEM_MAJOR_VERSION;
    *minor = ACLSHMEM_MINOR_VERSION;
}

void aclshmem_info_get_name(char* name)
{
    SHM_ASSERT_RET_VOID(name != nullptr);
    std::ostringstream oss;
    oss << "ACLSHMEM v" << ACLSHMEM_VENDOR_MAJOR_VER << "." << ACLSHMEM_VENDOR_MINOR_VER << "."
        << ACLSHMEM_VENDOR_PATCH_VER;
    auto version_str = oss.str();
    size_t i;
    for (i = 0; i < ACLSHMEM_MAX_NAME_LEN - 1 && version_str[i] != '\0'; i++) {
        name[i] = version_str[i];
    }
    name[i] = '\0';
}

int32_t aclshmemx_get_uniqueid(aclshmemx_uniqueid_t* uid)
{
    int status = aclshmemx_set_log_level(aclshmem_log::ERROR_LEVEL);

    ACLSHMEM_CHECK_RET(
        aclshmemi_bootstrap_pre_init(ACLSHMEMX_INIT_WITH_UNIQUEID, &g_boot_handle),
        "Get uniqueid failed during the bootstrap preloading step.");

    aclshmemx_uniqueid_t default_uid{};
    default_uid.version = ACLSHMEM_UNIQUEID_VERSION;
    *uid = default_uid;
    if (g_boot_handle.pre_init_ops) {
        ACLSHMEM_CHECK_RET(
            g_boot_handle.pre_init_ops->get_unique_id((void*)uid), "Get uniqueid failed during the get uniqueid step.");
    } else {
        SHM_LOG_ERROR("Pre_init_ops is empty, unique_id cannot be obtained.");
        status = ACLSHMEM_INVALID_PARAM;
    }
    return status;
}

int32_t aclshmemx_set_log_level(int level)
{
    // use env first, input level secondly, user may change level from env instead call func
    const char* in_level = std::getenv("SHMEM_LOG_LEVEL");
    if (in_level != nullptr) {
        auto tmp_level = std::string(in_level);
        if (tmp_level == "DEBUG") {
            level = aclshmem_log::DEBUG_LEVEL;
        } else if (tmp_level == "INFO") {
            level = aclshmem_log::INFO_LEVEL;
        } else if (tmp_level == "WARN") {
            level = aclshmem_log::WARN_LEVEL;
        } else if (tmp_level == "ERROR") {
            level = aclshmem_log::ERROR_LEVEL;
        } else if (tmp_level == "FATAL") {
            level = aclshmem_log::FATAL_LEVEL;
        }
    }

    return aclshmem_log::aclshmem_out_logger::Instance().set_log_level(static_cast<aclshmem_log::log_level>(level));
}

int32_t aclshmemx_set_conf_store_tls(bool enable, const char* tls_info, const uint32_t tls_info_len)
{
    g_boot_handle.tls_enable = enable;
    g_boot_handle.tls_info = tls_info;
    g_boot_handle.tls_info_len = tls_info_len;

    return ACLSHMEM_SUCCESS;
}

void aclshmem_rank_exit(int status)
{
    SHM_LOG_DEBUG("aclshmem_rank_exit is work ,status: " << status);
    exit(status);
}

int32_t aclshmemx_set_config_store_tls_key(
    const char* tls_pk, const uint32_t tls_pk_len, const char* tls_pk_pw, const uint32_t tls_pk_pw_len,
    const aclshmem_decrypt_handler decrypt_handler)
{
    g_boot_handle.tls_pk = tls_pk;
    g_boot_handle.tls_pk_len = tls_pk_len;
    g_boot_handle.tls_pk_pw = tls_pk_pw;
    g_boot_handle.tls_pk_pw_len = tls_pk_pw_len;
    g_boot_handle.decrypt_handler = decrypt_handler;

    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemx_set_extern_logger(void (*func)(int level, const char* msg))
{
    aclshmem_log::aclshmem_out_logger::Instance().set_extern_log_func(func);
    return ACLSHMEM_SUCCESS;
}

void aclshmem_global_exit(int status)
{
    if (g_boot_handle.is_bootstraped == true) {
        g_boot_handle.global_exit(status, &g_boot_handle);
    }
    SHM_LOG_WARN("Bootstrap not initialized. Global_exit Do nothing. ");
}

void aclshmemx_show_prof(aclshmem_prof_pe_t** out_profs, bool verbose)
{
    // 如果 verbose 为 false 且 out_profs 为 nullptr，直接返回
    if (!verbose && out_profs == nullptr) {
        return;
    }
    prof_data_print(&g_host_profs, &g_state, out_profs, verbose);
}
