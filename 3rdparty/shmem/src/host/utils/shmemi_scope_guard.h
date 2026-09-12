/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_SCOPE_GUARD_H
#define SHMEM_SCOPE_GUARD_H

#include <functional>
#include <type_traits>
#include <utility>
#include "shmemi_logger.h"
#include "shmemi_host_types.h"

namespace shm {
namespace utils {
template <typename T>
class scope_guard {
public:
    explicit scope_guard(T resource, std::function<void(T)> deleter)
        : deleter_(std::move(deleter)), resource_(std::move(resource)), owned_(true)
    {
        static_assert(std::is_nothrow_move_constructible_v<T>,
                      "T must be nothrow move constructible for exception safety");
    }

    // 禁止拷贝构造
    scope_guard(const scope_guard&) = delete;
    scope_guard &operator=(const scope_guard&) = delete;

    // 支持移动构造
    scope_guard(scope_guard &&other) noexcept
        : resource_(std::move(other.resource_)), deleter_(std::move(other.deleter_)), owned_(other.owned_)
    {
        other.owned_ = false; // other放弃所有权
    }
    scope_guard &operator=(scope_guard &&other) noexcept
    {
        if (this != &other) {
            if (owned_) {
                execute_deleter();
            }
            resource_ = std::move(other.resource_);
            deleter_ = std::move(other.deleter_);
            owned_ = other.owned_;
            other.owned_ = false;
        }
        return *this;
    }

    ~scope_guard() noexcept
    {
        if (owned_) {
            execute_deleter();
        }
    }

    // 放弃资源所有权
    void release() noexcept
    {
        owned_ = false;
    }

    // 手动立即释放资源
    void execute() noexcept
    {
        if (owned_) {
            execute_deleter();
            owned_ = false;
        }
    }

    // 放弃所有权并返回资源
    T abandon() noexcept
    {
        owned_ = false;
        return resource_;
    }

    // 获取托管的资源 
    T get() const noexcept
    {
        return resource_;
    }

    // 判断是否持有资源
    explicit operator bool() const noexcept
    {
        return owned_;
    }

private:
    void execute_deleter() noexcept
    {
        if (deleter_) {
            static_cast<void>(deleter_(resource_));
        }
    }

    // 声明顺序：deleter_ 必须在 resource_ 之前，确保构造时若 deleter_ 抛异常则 resource 尚未被移走
    std::function<void(T)> deleter_;
    T resource_;
    bool owned_{false};
};

template <typename T, typename D>
inline scope_guard<T> make_scope_guard(T resource, D deleter)
{
    return scope_guard<T>(std::move(resource), std::function<void(T)>(std::move(deleter)));
}

}  // namespace utils
}  // namespace shm

#endif // SHMEM_SCOPE_GUARD_H
