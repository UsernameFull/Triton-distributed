/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_SOCKET_H
#define SHMEM_SOCKET_H

#include "host/shmem_host_def.h"
#include "utils/shmemi_logger.h"
#include "utils/shmemi_host_types.h"
#include "init/bootstrap/shmemi_bootstrap.h"
#include "shmemi_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_IF_NAME_SIZE 16
#define SOCKET_TYPE_SEND 0
#define SOCKET_TYPE_RECV 1

#define RETRY_REFUSED_TIMES 1e5 // 100s超时
#define RETRY_TIMEDOUT_TIMES 50
#define SLEEP_INT 1000  // 重试间隔（微秒）

#define SOCKET_ACCEPT_TIMEOUT_MS 50000    // accept超时50秒
#define SOCKET_RECV_TIMEOUT_MS   30000    // recv超时30秒

#define SOCKET_BACKLOG 16384

typedef enum {
    // 初始状态：套接字刚创建，未执行任何操作
    SOCKET_STATE_CREATED,

    // 服务器端状态
    SOCKET_STATE_BOUND,         // 已绑定地址（bind 成功）
    SOCKET_STATE_LISTENING,     // 正在监听连接（listen 成功）
    SOCKET_STATE_ACCEPTING,     // 正在等待接受连接（准备调用 accept）
    SOCKET_STATE_ACCEPTED,      // 已接受连接（accept 成功，未完成验证）

    // 客户端状态
    SOCKET_STATE_CONNECTING,    // 正在发起连接（connect 调用中）
    SOCKET_STATE_CONNECTED,     // 已建立连接（未完成验证）

    // 公共状态
    SOCKET_STATE_READY,         // 连接已验证，可进行数据传输（最终就绪状态）
    SOCKET_STATE_ERROR,         // 发生错误
    SOCKET_STATE_CLOSED         // 已关闭
} socket_state_t;

typedef enum {
    SOCKET_TYPE_BOOTSTRAP,  // 用于初始化信息交换
    SOCKET_TYPE_DATA        // 用于实际数据传输
} socket_type_t;

typedef struct {
    int fd;                       // 套接字fd
    int accept_fd;                // 监听用fd，初始化-1
    sockaddr_t addr;              // 存储地址信息（socket_init阶段初始化）
    socket_state_t state;
    uint64_t magic;
    socket_type_t type;
    int refused_retries;       // 连接被拒绝重试计数
    int timeout_retries;       // 超时重试计数
} socket_t;

struct bootstrap_root_args {
    socket_t* listen_sock;
    uint64_t magic;
    int version;
};

// 其他内部结构体
typedef struct {
    int rank;
    int nranks;
    sockaddr_t ext_addr_listen;
    sockaddr_t ext_address_listen_root;
} bootstrap_ext_info;

struct bootstrap_netstate {
    char bootstrap_netifname[MAX_IF_NAME_SIZE + 1];     /* Socket Interface Name */
    sockaddr_t bootstrap_netifaddr; /* Socket Interface Address */
    int bootstrap_netinitdone = 0;                      /* Socket Interface Init Status */
    pthread_mutex_t bootstrap_netlock = PTHREAD_MUTEX_INITIALIZER; /* Socket Interface Lock */
    pthread_t bootstrap_root; /* Socket Root Thread for phoning root to non-root peers */
};

typedef struct unexpected_conn {
    int peer;                      // 发送方rank
    int tag;                       // 消息tag
    socket_t sock;                 // 对应的socket连接
    struct unexpected_conn* next;  // 链表下一个节点
} unexpected_conn_t;

typedef struct {
    int rank;
    int nranks;
    uint64_t magic;
    socket_t listen_sock;
    socket_t ring_send_sock;
    socket_t ring_recv_sock;
    sockaddr_t* peer_addrs;
    unexpected_conn_t* unexpected_conns;  // 意外连接队列
} uid_bootstrap_state;

int socket_init(socket_t* sock, socket_type_t type, uint64_t magic, const sockaddr_t* init_addr);
int socket_listen(socket_t* sock);
int socket_connect(socket_t* sock);
int socket_accept(socket_t* client_sock, socket_t* listen_sock);
int socket_send(socket_t* sock, void* ptr, int size);
int socket_recv(socket_t* sock, void* ptr, int size);
int socket_close(socket_t* sock);
int socket_get_sainfo(socket_t* sock, sockaddr* sa, socklen_t* addr_len);

#ifdef __cplusplus
}
#endif
#endif  // ACLSHMEM_SOCKET_H