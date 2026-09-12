/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_HOST_TYPES_H
#define SHMEMI_HOST_TYPES_H

#define ACLSHMEM_MAX_TRANSPORT_NUM 16

#include "host_device/shmem_common_types.h"
#define ACLSHMEM_MAX_HANDLE_IP_PORT_LEN 64
typedef struct aclshmemi_bootstrap_attr {
    aclshmemi_bootstrap_attr() : initialize_mf(0), mpi_comm(NULL), uid_args(NULL)
    {}
    int initialize_mf;
    void *mpi_comm;
    void *mete_data;
    void *uid_args;
} aclshmemi_bootstrap_attr_t;

typedef struct aclshmemi_bootstrap_init_ops {
    void *cookie;
    int (*get_unique_id)(void *cookit);
    int (*get_unique_id_static_magic)(void *uid_info, bool is_root);
} aclshmemi_bootstrap_init_ops_t;

typedef struct aclshmemi_bootstrap_handle {
    void        *bootstrap_state;
    aclshmemi_bootstrap_init_ops_t *pre_init_ops = nullptr;
    const char  *tls_info = nullptr;
    const char  *tls_pk;
    const char  *tls_pk_pw;
    
    int (*finalize)(aclshmemi_bootstrap_handle *boot_handle);
    int (*allgather)(const void *sendbuf, void *recvbuf, int size, aclshmemi_bootstrap_handle *boot_handle);
    int (*barrier)(aclshmemi_bootstrap_handle *boot_handle);
    int (*alltoall)(const void *sendbuf, void *recvbuf, int size, aclshmemi_bootstrap_handle *boot_handle);
    void (*global_exit)(int status, aclshmemi_bootstrap_handle *boot_handle);
    aclshmem_decrypt_handler decrypt_handler;

    int32_t     mype;
    int32_t     npes;
    int32_t     sockFd;
    int32_t     timeOut;
    int32_t     timeControlOut;
    uint32_t    tls_info_len;
    uint32_t    tls_pk_len;
    uint32_t    tls_pk_pw_len;

    bool is_bootstraped = false;
    bool use_attr_ipport = false;
    bool tls_enable = false;

    char ipport[ACLSHMEM_MAX_HANDLE_IP_PORT_LEN];
} aclshmemi_bootstrap_handle_t;

typedef struct aclshmemi_bootstrap_mpi_options {
    // TBD
} aclshmemi_bootstrap_mpi_options_t;

typedef struct aclshmemi_bootstrap_uid_options {
    // TBD
} aclshmemi_bootstrap_uid_options_t;

typedef struct aclshmemi_transport_pe_info {
    int32_t     pe;
    int32_t     dev_id;
    int64_t     server_id;
    int64_t     superpod_id;
} aclshmemi_transport_pe_info_t;

typedef struct aclshmemi_transport {
    // control plane
    int (*can_access_peer)(int *access, aclshmemi_transport_pe_info_t *peer_info,
                            aclshmemi_transport_pe_info_t *my_info, struct aclshmemi_transport *t);
    int (*connect_peers)(struct aclshmemi_transport *t, int *selected_dev_ids,
                            int num_selected_devs, aclshmem_device_host_state_t *g_state);
    int (*finalize)(struct aclshmemi_transport *t,
                        aclshmem_device_host_state_t *g_state);

    // data plane, TBD
    void (*rma)(struct aclshmemi_transport *t, int32_t type, void *dst, void *src, size_t size, int32_t pe);
    void (*amo)(struct aclshmemi_transport *t, int32_t type, void *dst, void *src, size_t size, int32_t pe);
    void (*quiet)(struct aclshmemi_transport *t);
    void (*fence)(struct aclshmemi_transport *t);
    int32_t     logical_dev_id;
    int32_t     dev_id;
} aclshmemi_transport_t;

typedef struct {
    int32_t pe, npes;

    aclshmemi_bootstrap_mpi_options_t mpi_options;
    aclshmemi_bootstrap_uid_options_t uid_options;
    
    // other options
    bool rdma_enabled;
} aclshmemi_options_t;

// host only state
typedef struct {
    // typedef void *aclrtStream; as in https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/appdevgapi/aclcppdevg_03_1355.html
    void *default_stream;
    // using TEventID = int8_t; as in https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/ascendcopapi/atlasascendc_api_07_0181.html
    int8_t default_event_id;
    uint32_t default_block_num;

    // topo
    int32_t *transport_map;                 /* npes * npes, 2D-Array, point-to-connectivity. */
    aclshmemi_transport_pe_info *pe_info;      /* All pe's host info, need to build transports. */

    aclshmemi_options_t options;

    aclshmemi_bootstrap_handle_t *boot_handle;
    aclshmemi_transport_t choosen_transports[ACLSHMEM_MAX_TRANSPORT_NUM];
    int32_t num_choosen_transport;
} aclshmemi_host_trans_state_t;

#endif // ACLSHMEMI_HOST_TYPES_H