/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <fcntl.h>
#include <netinet/tcp.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <poll.h>
#include <algorithm>
#include "uid_socket.h"

static int socket_poll_fd(int fd, int events, int timeout_ms) {
    struct pollfd pfd = {0};
    pfd.fd = fd;
    pfd.events = events;

    int ret = poll(&pfd, 1, timeout_ms);
    if (ret == -1) {
        SHM_LOG_ERROR("poll failed: " << strerror(errno) << " (fd: " << fd << ")");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    } else if (ret == 0) {
        SHM_LOG_ERROR("poll timeout (" << timeout_ms << "ms) - fd: " << fd);
        return ACLSHMEM_TIMEOUT_ERROR;
    }

    // 检查fd错误
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
        SHM_LOG_ERROR("fd error (revents: " << pfd.revents << ") - fd: " << fd);
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    return ACLSHMEM_SUCCESS;
}

static int socket_progress(int op, socket_t* sock, void* ptr, int size, int* offset, bool block = false, bool state_check = true) {
    if (sock == nullptr || ptr == nullptr || offset == nullptr || size < 0 || *offset < 0 || *offset > size) {
        SHM_LOG_ERROR("Invalid arguments: sock=" << sock << ", ptr=" << ptr 
                  << ", size=" << size << ", offset=" << *offset);
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    if (state_check && sock->state != SOCKET_STATE_READY) {
        SHM_LOG_ERROR("socket_progress: invalid state " << sock->state << " (expected READY)");
        sock->state = SOCKET_STATE_ERROR;
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    int poll_events = (op == SOCKET_TYPE_RECV) ? POLLIN : POLLOUT;
    ACLSHMEM_CHECK_RET(socket_poll_fd(sock->fd, poll_events, SOCKET_RECV_TIMEOUT_MS), "socket_poll_fd failed.");

    int bytes = 0;
    int closed = 0;
    char* data = (char*)(ptr);
    SHM_LOG_DEBUG("socket_progress: start");
    do {
        if (op == SOCKET_TYPE_RECV) {
            int flags = block ? 0 : MSG_DONTWAIT;
            SHM_LOG_DEBUG("Executing RECV operation - fd: " << sock->fd << ", buffer offset: " << *offset << ", remaining size: " << (size - *offset) << ", flags: " << flags);
            bytes = recv(sock->fd, data + *offset, size - *offset, flags);
            SHM_LOG_DEBUG("RECV result - bytes received: " << bytes);
        } else if (op == SOCKET_TYPE_SEND) {
            int flags = block ? MSG_NOSIGNAL : (MSG_DONTWAIT | MSG_NOSIGNAL);
            SHM_LOG_DEBUG("Executing SEND operation - fd: " << sock->fd << ", buffer offset: " << *offset << ", remaining size: " << (size - *offset) << ", flags: " << flags);
            bytes = send(sock->fd, data + *offset, size - *offset, flags);
            SHM_LOG_DEBUG("SEND result - bytes sent: " << bytes);
        } else {
            SHM_LOG_ERROR("Invalid operation type: " << op);
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }

        if (op == SOCKET_TYPE_RECV && bytes == 0) {
            SHM_LOG_DEBUG("RECV operation got 0 bytes - remote peer closed the connection (fd: " << sock->fd << ")");
            closed = 1;
            break;
        }

        if (bytes == -1) {
            int err = errno;
            if (err != EINTR && err != EWOULDBLOCK && err != EAGAIN) {
                SHM_LOG_ERROR("Socket operation failed (fd: " << sock->fd << ", op: " << op << ") - error: " << strerror(err) << " (errno: " << err << ")");
                return ACLSHMEM_BOOTSTRAP_ERROR;
            } else {
                SHM_LOG_DEBUG("Socket operation would block (fd: " << sock->fd << ", op: " << op << ") - errno: " << err << ", setting bytes to 0");
                bytes = 0;
            }
        }

        *offset += bytes;
        SHM_LOG_DEBUG("Updated buffer offset - current offset: " << *offset << ", total size: " << size);
    } while (bytes > 0 && *offset < size);

    if (closed) {
        SHM_LOG_ERROR("Loop exited - remote connection closed (fd: " << sock->fd << ")");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    SHM_LOG_DEBUG("socket_progress: success");

    return ACLSHMEM_SUCCESS;
}

static int socket_wait(int op, socket_t* sock, void* ptr, int size, int* offset, bool block = false, bool state_check = true) {
    while (*offset < size)
        if (socket_progress(op, sock, ptr, size, offset, block, state_check) != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("socket_wait fail!");
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }
    return ACLSHMEM_SUCCESS;
}

int socket_send(socket_t* sock, void* ptr, int size) {
    SHM_LOG_DEBUG("socket_send: start");
    int offset = 0;
    if (sock == NULL || ptr == NULL || size <= 0 ) {
        SHM_LOG_ERROR("send sock == NULL");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
  
  return socket_wait(SOCKET_TYPE_SEND, sock, ptr, size, &offset);
}

int socket_recv(socket_t* sock, void* ptr, int size) {
    SHM_LOG_DEBUG("socket_recv: start");
    int offset = 0;
    if (sock == NULL) {
        SHM_LOG_ERROR("recv sock == NULL");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
  return socket_wait(SOCKET_TYPE_RECV, sock, ptr, size, &offset);
}


int socket_close(socket_t* sock) {
    if (sock) {
        if (sock->fd >= 0) {
            shutdown(sock->fd, SHUT_RDWR);
            close(sock->fd);
        }
        sock->fd = -1;
        sock->accept_fd = -1;
        sock->state = SOCKET_STATE_CLOSED;
    } else {
        SHM_LOG_DEBUG("socket_close: sock is null");
    }
    SHM_LOG_DEBUG("socket_close: success");
    return ACLSHMEM_SUCCESS;
}

int socket_get_sainfo(socket_t* sock, sockaddr* sa, socklen_t* addr_len) {
    if (sock == nullptr || sa == nullptr || addr_len == nullptr) {
        SHM_LOG_ERROR("Some of sock, sa and addr_len are null.");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    
    if (sock->addr.type == ADDR_IPv4) {
        SHM_LOG_DEBUG("socket_get_sainfo memcpy addr4");
        std::copy_n(reinterpret_cast<const char*>(&sock->addr.addr.addr4), 
                    sizeof(struct sockaddr_in), 
                    reinterpret_cast<char*>(sa));
        *addr_len = sizeof(struct sockaddr_in);
    } else {
        SHM_LOG_DEBUG("socket_get_sainfo memcpy addr6");
        std::copy_n(reinterpret_cast<const char*>(&sock->addr.addr.addr6), 
                    sizeof(struct sockaddr_in6), 
                    reinterpret_cast<char*>(sa));
        *addr_len = sizeof(struct sockaddr_in6);
    }

    return ACLSHMEM_SUCCESS;
}


int socket_listen(socket_t* sock) {
    if (!sock || sock->fd < 0 || sock->state == SOCKET_STATE_ERROR) {
        SHM_LOG_ERROR("socket_listen Precondition failed! "
                  << "sock is null: " << (sock == nullptr)
                  << ", invalid fd: " << (sock ? (sock->fd < 0) : true)
                  << ", state is error: " << (sock ? (sock->state == SOCKET_STATE_ERROR) : false));
        if (sock) sock->state = SOCKET_STATE_ERROR;
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    SHM_LOG_INFO("socket_listen Entering. sock fd: " << (sock ? sock->fd : -1)
              << ", current state: " << (sock ? sock->state : -1));

    if (sock->state == SOCKET_STATE_CREATED) {
        SHM_LOG_DEBUG("socket_listen State is CREATED, starting bind process");
        struct sockaddr_storage sa_storage;
        std::fill(reinterpret_cast<char*>(&sa_storage), 
                  reinterpret_cast<char*>(&sa_storage) + sizeof(sa_storage), 
                  0);
        struct sockaddr* sa = reinterpret_cast<struct sockaddr*>(&sa_storage);
        socklen_t addr_len;

        ACLSHMEM_CHECK_RET(socket_get_sainfo(sock, sa, &addr_len),"socket_listen socket_get_sainfo failed");
        

        std::string target_ip = "unknown";
        uint16_t target_port = 0;
        if (sa->sa_family == AF_INET) {
            struct sockaddr_in* ipv4 = reinterpret_cast<struct sockaddr_in*>(sa);
            char ip_str[INET_ADDRSTRLEN] = {0};
            ACLSHMEM_CHECK_RET(inet_ntop(AF_INET, &ipv4->sin_addr, ip_str, sizeof(ip_str)) == nullptr, "convert ipv4 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
            target_ip = ip_str;
            target_port = ntohs(ipv4->sin_port);
        } else if (sa->sa_family == AF_INET6) {
            struct sockaddr_in6* ipv6 = reinterpret_cast<struct sockaddr_in6*>(sa);
            char ip_str[INET6_ADDRSTRLEN] = {0};
            ACLSHMEM_CHECK_RET(inet_ntop(AF_INET6, &ipv6->sin6_addr, ip_str, sizeof(ip_str)) == nullptr, "convert ipv6 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
            target_ip = ip_str;
            target_port = ntohs(ipv6->sin6_port);
        }
        SHM_LOG_DEBUG("socket_listen socket_get_sainfo succeeded, addr_len: " << addr_len
                  << ", target IP: " << target_ip << ", target port: " << target_port);

        int opt = 1;
        if (setsockopt(sock->fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            SHM_LOG_ERROR("socket_listen setsockopt(SO_REUSEADDR) failed! "
                      << "errno: " << errno << ", reason: " << strerror(errno));
            sock->state = SOCKET_STATE_ERROR;
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }
        SHM_LOG_DEBUG("socket_listen setsockopt(SO_REUSEADDR) succeeded");

        if (bind(sock->fd, sa, addr_len) < 0) {
            SHM_LOG_ERROR("socket_listen bind failed! "
                      << "errno: " << errno << ", reason: " << strerror(errno)
                      << ", fd: " << sock->fd << ", addr_len: " << addr_len
                      << ", target IP: " << target_ip << ", target port: " << target_port);
            sock->state = SOCKET_STATE_ERROR;
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }
        SHM_LOG_DEBUG("[socket_listen] bind succeeded");

        if (getsockname(sock->fd, &sock->addr.addr.sa, &addr_len) < 0) {
            SHM_LOG_ERROR("socket_listen getsockname failed! "
                      << "errno: " << errno << ", reason: " << strerror(errno));
            sock->state = SOCKET_STATE_ERROR;
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }

        if (sock->addr.type == ADDR_IPv4) {
            struct sockaddr_in* ipv4 = &sock->addr.addr.addr4;
            char ip_str[INET_ADDRSTRLEN] = {0};
            ACLSHMEM_CHECK_RET(inet_ntop(AF_INET, &ipv4->sin_addr, ip_str, sizeof(ip_str)) == nullptr, "convert ipv4 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
            uint16_t port = ntohs(ipv4->sin_port);
            SHM_LOG_DEBUG(" Stored IPv4 address: " << ip_str << ":" << port << " sa_family: " << ipv4->sin_family << " (expected AF_INET=" << AF_INET << ")");
        } else if (sock->addr.type == ADDR_IPv6) {
            struct sockaddr_in6* ipv6 = &sock->addr.addr.addr6;
            char ip_str[INET6_ADDRSTRLEN] = {0};
            ACLSHMEM_CHECK_RET(inet_ntop(AF_INET6, &ipv6->sin6_addr, ip_str, sizeof(ip_str)) == nullptr, "convert ipv6 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
            uint16_t port = ntohs(ipv6->sin6_port);
            SHM_LOG_DEBUG(" Stored IPv6 address: " << ip_str << ":" << port << " sa_family: " << ipv6->sin6_family << " (expected AF_INET6=" << AF_INET6 << ")");
        } else {
            SHM_LOG_ERROR(" Stored address type: unknown (type=" << sock->addr.type << ")");
            sock->state = SOCKET_STATE_ERROR;
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }

        sock->state = SOCKET_STATE_BOUND;
        SHM_LOG_DEBUG("socket_listen State updated to BOUND");
    }

    if (sock->state == SOCKET_STATE_BOUND) {
        SHM_LOG_DEBUG("socket_listen State is BOUND, starting listen");
        if (listen(sock->fd, SOCKET_BACKLOG) < 0) {
            SHM_LOG_ERROR("socket_listen] listen failed! "
                      << "errno: " << errno << ", reason: " << strerror(errno)
                      << ", fd: " << sock->fd << ", backlog: " << SOCKET_BACKLOG);
            sock->state = SOCKET_STATE_ERROR;
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }
        sock->accept_fd = sock->fd;
        sock->state = SOCKET_STATE_LISTENING;
        SHM_LOG_DEBUG("socket_listen listen succeeded. New state: LISTENING, accept_fd: " << sock->accept_fd);
    } else {
        SHM_LOG_ERROR("socket_listen Skip listen: current state is " << sock->state << " (expected BOUND)");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    SHM_LOG_DEBUG("socket_listen Exiting with success");
    return ACLSHMEM_SUCCESS;
}

static int socket_try_accept(socket_t* sock) {
    if (sock->state != SOCKET_STATE_ACCEPTING) {
        SHM_LOG_ERROR("socket_try_accept: invalid state " << sock->state);
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    ACLSHMEM_CHECK_RET(socket_poll_fd(sock->accept_fd, POLLIN, SOCKET_ACCEPT_TIMEOUT_MS), "socket_poll_fd failed.");
    struct sockaddr sa;
    socklen_t socklen = sizeof(sa);

    sock->fd = accept(sock->accept_fd, &sa, &socklen);
    if (sock->fd != -1) {
        if (sa.sa_family == AF_INET) {
            sock->addr.type = ADDR_IPv4;
            std::copy_n(reinterpret_cast<const char*>(&sa), 
                        sizeof(struct sockaddr_in), 
                        reinterpret_cast<char*>(&sock->addr.addr.addr4));
        } else {
            sock->addr.type = ADDR_IPv6;
            std::copy_n(reinterpret_cast<const char*>(&sa), 
                        sizeof(struct sockaddr_in6), 
                        reinterpret_cast<char*>(&sock->addr.addr.addr6));
        }
        sock->state = SOCKET_STATE_ACCEPTED;
    } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
        SHM_LOG_ERROR("socket_try_accept failed: " << strerror(errno));
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    return ACLSHMEM_SUCCESS;
}

static int socket_finalize_accept(socket_t* sock) {
    if (sock->state != SOCKET_STATE_ACCEPTED) {
        SHM_LOG_ERROR("socket_finalize_accept: invalid state " << sock->state);
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    uint64_t magic;
    socket_type_t type;
    int received = 0;
    const int one = 1;

    if (setsockopt(sock->fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) < 0) {
        SHM_LOG_ERROR("setsockopt TCP_NODELAY failed: " << strerror(errno));
        close(sock->fd);
        sock->fd = -1;
        sock->state = SOCKET_STATE_ERROR;
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    if (socket_progress(SOCKET_TYPE_RECV, sock, &magic, sizeof(magic), &received, false, false) != ACLSHMEM_SUCCESS) {
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    if (received == 0) return ACLSHMEM_SUCCESS;
    if (socket_wait(SOCKET_TYPE_RECV, sock, &magic, sizeof(magic), &received, false, false) != ACLSHMEM_SUCCESS) {
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    if (magic != sock->magic) {
        SHM_LOG_DEBUG("socket_finalize_accept: wrong magic " << magic << " != " << sock->magic);
        close(sock->fd);
        sock->fd = -1;
        sock->state = SOCKET_STATE_ACCEPTING;
        return ACLSHMEM_SUCCESS;
    }

    received = 0;
    if (socket_wait(SOCKET_TYPE_RECV, sock, &type, sizeof(type), &received, false, false) != ACLSHMEM_SUCCESS) {
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    if (type != sock->type) {
        SHM_LOG_ERROR("socket_finalize_accept: wrong type " << type << " != " << sock->type);
        close(sock->fd);
        sock->fd = -1;
        sock->state = SOCKET_STATE_ERROR;
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    sock->state = SOCKET_STATE_READY;
    return ACLSHMEM_SUCCESS;
}

static int socket_start_connect(socket_t* sock) {
    if (sock->state != SOCKET_STATE_CONNECTING) {
        SHM_LOG_ERROR("socket_start_connect: invalid state " << sock->state);
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    struct sockaddr_storage sa_storage;
    std::fill(reinterpret_cast<char*>(&sa_storage), 
              reinterpret_cast<char*>(&sa_storage) + sizeof(sa_storage), 
              0);
    struct sockaddr* sa = reinterpret_cast<struct sockaddr*>(&sa_storage);
    socklen_t addr_len;
    if (socket_get_sainfo(sock, sa, &addr_len) != 0) {
        sock->state = SOCKET_STATE_ERROR;
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    int ret = connect(sock->fd, sa, addr_len);
    if (ret == 0) {
        sock->state = SOCKET_STATE_CONNECTED;
        SHM_LOG_DEBUG("socket_start_connect: success!");
    } else if (errno == ECONNREFUSED) {
        SHM_LOG_DEBUG("socket_start_connect: refused retry time:" << sock->refused_retries);
        if (++sock->refused_retries >= RETRY_REFUSED_TIMES) {
            SHM_LOG_ERROR("exceeded refused retries");
            sock->state = SOCKET_STATE_ERROR;
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }
        usleep(SLEEP_INT);
    } else if (errno == ETIMEDOUT) {
        SHM_LOG_DEBUG("socket_start_connect: timeout retry time:" << sock->timeout_retries);
        if (++sock->timeout_retries >= RETRY_TIMEDOUT_TIMES) {
            SHM_LOG_ERROR("exceeded timeout retries");
            sock->state = SOCKET_STATE_ERROR;
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }
        usleep(SLEEP_INT);
    } else {
        SHM_LOG_ERROR("connect failed: " << strerror(errno));
        sock->state = SOCKET_STATE_ERROR;
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    SHM_LOG_DEBUG("socket_start_connect: end!");

    return ACLSHMEM_SUCCESS;
}


static int socket_finalize_connect(socket_t* sock) {
    SHM_LOG_DEBUG("socket_finalize_connect socket_finalize_connect: start!");
    if (sock->state != SOCKET_STATE_CONNECTED) {
        SHM_LOG_ERROR("socket_finalize_connect: invalid state " << sock->state);
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    int sent = 0;
    if (socket_progress(SOCKET_TYPE_SEND, sock, &sock->magic, sizeof(sock->magic), &sent, false, false) != ACLSHMEM_SUCCESS) {
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    if (sent == 0) return ACLSHMEM_SUCCESS;
    if (socket_wait(SOCKET_TYPE_SEND, sock, &sock->magic, sizeof(sock->magic), &sent, false, false) != ACLSHMEM_SUCCESS) {
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    sent = 0;
    if (socket_wait(SOCKET_TYPE_SEND, sock, &sock->type, sizeof(sock->type), &sent, false, false) != ACLSHMEM_SUCCESS) {
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    SHM_LOG_DEBUG("socket_finalize_connect socket_finalize_connect: end!");

    sock->state = SOCKET_STATE_READY;
    return ACLSHMEM_SUCCESS;
}

static int socket_progress_state(socket_t* sock) {
    if (sock == nullptr) {
        SHM_LOG_ERROR("socket_progress_state: null socket");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    if (sock->state == SOCKET_STATE_ACCEPTING) {
        ACLSHMEM_CHECK_RET(socket_try_accept(sock), "socket_try_accept failed");
    }
    if (sock->state == SOCKET_STATE_ACCEPTED) {
        ACLSHMEM_CHECK_RET(socket_finalize_accept(sock), "socket_finalize_accept failed");
    }
    if (sock->state == SOCKET_STATE_CONNECTING) {
        ACLSHMEM_CHECK_RET(socket_start_connect(sock), "socket_start_connect failed");
    }

    if (sock->state == SOCKET_STATE_CONNECTED) {
        ACLSHMEM_CHECK_RET(socket_finalize_connect(sock), "socket_finalize_connect failed");
    }

    return ACLSHMEM_SUCCESS;
}

int socket_connect(socket_t* sock) {
    if (sock == nullptr) {
        SHM_LOG_ERROR("socket_connect: NULL socket");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    if (sock->fd == -1) {
        SHM_LOG_ERROR("socket_connect: invalid fd (-1)");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    if (sock->state != SOCKET_STATE_CREATED) {
        SHM_LOG_ERROR("socket_connect: invalid state " << sock->state << " (expected CREATED)");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    const int one = 1;
    // Disabling the Nagle algorithm
    if (setsockopt(sock->fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) < 0) {
        SHM_LOG_ERROR("setsockopt TCP_NODELAY failed: " << strerror(errno));
        sock->state = SOCKET_STATE_ERROR;
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    sock->state = SOCKET_STATE_CONNECTING;
    SHM_LOG_DEBUG("socket_connect: start!");
    do {
        if (socket_progress_state(sock) != ACLSHMEM_SUCCESS) {
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }
    } while (sock->state == SOCKET_STATE_CONNECTING ||
              sock->state == SOCKET_STATE_CONNECTED);

    switch (sock->state) {
        case SOCKET_STATE_READY:
            return ACLSHMEM_SUCCESS;
        case SOCKET_STATE_ERROR:
            return ACLSHMEM_BOOTSTRAP_ERROR;
        default:
            return ACLSHMEM_BOOTSTRAP_ERROR;
    }
}

int socket_accept(socket_t* client_sock, socket_t* listen_sock) {
    if (listen_sock == nullptr || client_sock == nullptr) {
        SHM_LOG_ERROR("socket_accept: NULL socket");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    if (listen_sock->state != SOCKET_STATE_LISTENING) {
        SHM_LOG_ERROR("socket_accept: listen socket state " << listen_sock->state << " (expected LISTENING)");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    if (client_sock->accept_fd == -1) {
        client_sock->addr = listen_sock->addr;
        client_sock->magic = listen_sock->magic;
        client_sock->type = listen_sock->type;
        client_sock->refused_retries = 0;
        client_sock->timeout_retries = 0;
        client_sock->accept_fd = listen_sock->fd;
        client_sock->fd = -1;
        client_sock->state = SOCKET_STATE_ACCEPTING;
    }
    SHM_LOG_DEBUG("socket_accept: start!");
    do {
        if (socket_progress_state(client_sock) != ACLSHMEM_SUCCESS) {
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }
    } while (client_sock->state == SOCKET_STATE_ACCEPTING ||
             client_sock->state == SOCKET_STATE_ACCEPTED);

    switch (client_sock->state) {
        case SOCKET_STATE_READY:
            return ACLSHMEM_SUCCESS;
        case SOCKET_STATE_ERROR:
            return ACLSHMEM_BOOTSTRAP_ERROR;
        default:
            return ACLSHMEM_BOOTSTRAP_ERROR;
    }
}

int socket_init(socket_t* sock, socket_type_t type, uint64_t magic, const sockaddr_t* init_addr) {
    if (sock == nullptr) {
        SHM_LOG_ERROR("socket_init: NULL socket");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    SHM_LOG_DEBUG("socket_init: start");
    std::fill(reinterpret_cast<char*>(sock), 
              reinterpret_cast<char*>(sock) + sizeof(socket_t), 
              0);
    sock->fd = -1;
    sock->accept_fd = -1;
    sock->state = SOCKET_STATE_CREATED;
    sock->type = type;
    sock->magic = magic;
    sock->refused_retries = 0;
    sock->timeout_retries = 0;

    if (init_addr != nullptr) {
        int family;
        if (init_addr->type == ADDR_IPv4) {
            family = AF_INET;
            std::copy_n(reinterpret_cast<const char*>(&init_addr->addr.addr4), 
                        sizeof(struct sockaddr_in), 
                        reinterpret_cast<char*>(&sock->addr.addr.addr4));
        } else if (init_addr->type == ADDR_IPv6) {
            family = AF_INET6;
            std::copy_n(reinterpret_cast<const char*>(&init_addr->addr.addr6), 
                        sizeof(struct sockaddr_in6), 
                        reinterpret_cast<char*>(&sock->addr.addr.addr6));
        } else {
            SHM_LOG_ERROR("socket_init: unsupported address type " << init_addr->type);
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }
        sock->addr.type = init_addr->type;

        sock->fd = socket(family, SOCK_STREAM, 0);
        if (sock->fd == -1) {
            SHM_LOG_ERROR("socket_init: create socket failed: " << strerror(errno));
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }
    } else {
        SHM_LOG_DEBUG("socket_init: init_addr is null");
        std::fill(reinterpret_cast<char*>(&sock->addr), 
                  reinterpret_cast<char*>(&sock->addr) + sizeof(sock->addr), 
                  0);
        sock->addr.type = ADDR_IPv4;
    }

    // set blocking
    if (sock->fd >= 0) {
        int32_t value = 1;
        if ((value = fcntl(sock->fd, F_GETFL)) == -1) {
            SHM_LOG_ERROR("sock: " << sock->fd <<" failed to get control value");
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }
        int new_flags = value & ~O_NONBLOCK;
        if (fcntl(sock->fd, F_SETFL, new_flags) == -1) {
            SHM_LOG_ERROR("sock: " << sock->fd << "Failed to set control value of link");
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }
    }
    
    SHM_LOG_DEBUG("socket_init: success");
    return ACLSHMEM_SUCCESS;
}