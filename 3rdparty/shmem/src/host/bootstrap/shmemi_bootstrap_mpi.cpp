/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cassert>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include "host/shmem_host_def.h"
#include "utils/shmemi_logger.h"
#include "utils/shmemi_host_types.h"
#include "init/bootstrap/shmemi_bootstrap.h"

typedef struct {
    MPI_Comm comm;
    int mpi_initialized;
} aclshmemi_bootstrap_mpi_state_t;
static aclshmemi_bootstrap_mpi_state_t aclshmemi_bootstrap_mpi_state = {MPI_COMM_NULL, 0};

static int aclshmemi_bootstrap_mpi_barrier(aclshmemi_bootstrap_handle_t *handle) {
    int status = MPI_SUCCESS;

    status = MPI_Barrier(aclshmemi_bootstrap_mpi_state.comm);
    ACLSHMEM_CHECK_RET(status);

    return status;
}

static int aclshmemi_bootstrap_mpi_allgather(const void *sendbuf, void *recvbuf, int length,
                                   aclshmemi_bootstrap_handle_t *handle) {
    int status = MPI_SUCCESS;

    status = MPI_Allgather(sendbuf, length, MPI_BYTE, recvbuf, length, MPI_BYTE, aclshmemi_bootstrap_mpi_state.comm);
    ACLSHMEM_CHECK_RET(status);

    return status;
}

static int aclshmemi_bootstrap_mpi_alltoall(const void *sendbuf, void *recvbuf, int length,
                                  aclshmemi_bootstrap_handle_t *handle) {
    int status = MPI_SUCCESS;

    status = MPI_Alltoall(sendbuf, length, MPI_BYTE, recvbuf, length, MPI_BYTE, aclshmemi_bootstrap_mpi_state.comm);
    ACLSHMEM_CHECK_RET(status);

    return status;
}

static void aclshmemi_bootstrap_mpi_global_exit(int status, aclshmemi_bootstrap_handle_t *handle) {
    int rc = MPI_SUCCESS;

    rc = MPI_Abort(aclshmemi_bootstrap_mpi_state.comm, status);
    if (rc != MPI_SUCCESS) {
        exit(1);
    }
}

static int aclshmemi_bootstrap_mpi_finalize(aclshmemi_bootstrap_handle_t *handle) {
    int status = MPI_SUCCESS, finalized;

    status = MPI_Finalized(&finalized);
    ACLSHMEM_CHECK_RET(status);

    if (finalized) {
        if (aclshmemi_bootstrap_mpi_state.mpi_initialized) {
            status = ACLSHMEM_INNER_ERROR;
        } else {
            status = 0;
        }

        return status;
    }

    if (!finalized && aclshmemi_bootstrap_mpi_state.mpi_initialized) {
        status = MPI_Comm_free(&aclshmemi_bootstrap_mpi_state.comm);
        ACLSHMEM_CHECK_RET(status);
    }

    if (aclshmemi_bootstrap_mpi_state.mpi_initialized) MPI_Finalize();

    return status;
}

int aclshmemi_bootstrap_plugin_init(void *mpi_comm, aclshmemi_bootstrap_handle_t *handle) {
    int status = MPI_SUCCESS, initialized = 0, finalized = 0;
    MPI_Comm src_comm;
    if (NULL == mpi_comm)
        src_comm = MPI_COMM_WORLD;
    else
        src_comm = *((MPI_Comm *)mpi_comm);
    status = MPI_Initialized(&initialized);
    ACLSHMEM_CHECK_RET(status);
    status = MPI_Finalized(&finalized);
    ACLSHMEM_CHECK_RET(status);
    if (!initialized && !finalized) {
        MPI_Init(NULL, NULL);
        aclshmemi_bootstrap_mpi_state.mpi_initialized = 1;

        if (src_comm != MPI_COMM_WORLD && src_comm != MPI_COMM_SELF) {
            status = ACLSHMEM_INNER_ERROR;
            if (aclshmemi_bootstrap_mpi_state.mpi_initialized) {
                MPI_Finalize();
                aclshmemi_bootstrap_mpi_state.mpi_initialized = 0;
            } 
        }
    } else if (finalized) {
        status = ACLSHMEM_INNER_ERROR;
        if (aclshmemi_bootstrap_mpi_state.mpi_initialized) {
            MPI_Finalize();
            aclshmemi_bootstrap_mpi_state.mpi_initialized = 0;
        }
    }
    status = MPI_Comm_dup(src_comm, &aclshmemi_bootstrap_mpi_state.comm);
    ACLSHMEM_CHECK_RET(status);
    status = MPI_Comm_rank(aclshmemi_bootstrap_mpi_state.comm, &handle->mype);
    ACLSHMEM_CHECK_RET(status);
    status = MPI_Comm_size(aclshmemi_bootstrap_mpi_state.comm, &handle->npes);
    ACLSHMEM_CHECK_RET(status);
    handle->allgather = aclshmemi_bootstrap_mpi_allgather;
    handle->alltoall = aclshmemi_bootstrap_mpi_alltoall;
    handle->barrier = aclshmemi_bootstrap_mpi_barrier;
    handle->global_exit = aclshmemi_bootstrap_mpi_global_exit;
    handle->finalize = aclshmemi_bootstrap_mpi_finalize;
    handle->pre_init_ops = NULL;
    handle->bootstrap_state = &aclshmemi_bootstrap_mpi_state.comm;
    return status;
}

