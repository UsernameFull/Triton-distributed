# 多实例支持

## 概述

SHMEM 多实例能力允许**在单个进程内创建多个独立 SHMEM 实例**。每个实例拥有独立的通信域、内存空间和同步资源，适用于以下场景：

- **多通信域共存**：单个进程中多个任务域需要独立 SHMEM 上下文
- **资源隔离**：不同任务/模块之间隔离 SHMEM 资源，避免相互干扰
- **弹性扩缩**：不同实例创建/释放互不影响
- **动态实例管理**：运行时按 `instance_id` 创建/释放实例

## 核心概念

### 实例（Instance）
一个 SHMEM 实例包含一组完整初始化状态，典型包括：
- 独立的通信域（`Team World`）
- 独立的共享内存堆（`Heap`）
- 独立的同步资源（`Sync Pool/Counter`）
- 独立的 State/Bootstrap/MemoryManager 运行时快照

### 实例标识（Instance ID）
当前仓库使用 `instance_id`（`uint64_t`）标识实例。实例生命周期由初始化属性和 `finalize` 接口管理。

### Context 切换
SHMEM 通过**全局变量 swap 机制**完成实例切换：切换时先回写当前实例的全局状态，再装载目标实例的全局状态。

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                     单进程 (Process)                     │
│  ┌───────────────────────────────────────────────────┐  │
│  │              SHMEM Runtime (全局)                  │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │  │
│  │  │  Instance 0 │  │  Instance 1 │  │  Instance N│ │  │
│  │  │  ┌───────┐  │  │  ┌───────┐  │  │  ┌──────┐ │  │  │
│  │  │  │ Team  │  │  │  │ Team  │  │  │  │ Team │ │  │  │
│  │  │  │ Heap  │  │  │  │ Heap  │  │  │  │ Heap │ │  │  │
│  │  │  │ Sync  │  │  │  │ Sync  │  │  │  │ Sync │ │  │  │
│  │  │  │ State │  │  │  │ State │  │  │  │ State│ │  │  │
│  │  │  └───────┘  │  │  └───────┘  │  │  └──────┘ │  │  │
│  │  └─────────────┘  └─────────────┘  └───────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 关键设计决策

| 设计点 | 当前实现方案 | 说明 |
|---|---|---|
| 实例隔离 | 独立 `aclshmem_context` 快照 | 每个实例保存独立的 `state`、`host_state`、`boot_handle`、`memory_manager` 等 |
| Context 管理 | 全局变量 swap | 通过 `aclshmemx_instance_ctx_set` 切换 |
| 实例标识 | `instance_id` (`uint64_t`) | 位于 `aclshmemx_init_attr_t`，并用于 `aclshmem_finalize(instance_id)` |
| Device 侧路由 | `instance_id << 1` | kernel 侧通过 硬编码地址 访问对应实例的 `state` |

## 快速上手（当前仓库 API）

### 1. 创建实例

```cpp
#include "shmem.h"

aclshmemx_init_attr_t attr;
attr.my_pe = pe_id;
attr.n_pes = pe_size;
attr.local_mem_size = 1024UL * 1024UL * 1024UL;
attr.ip_port = "tcp://x.x.x.x:0"        // 预先置为0,后续会分配
attr.comm_args = nullptr;               // default mode
attr.instance_id = 2;

int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
```

### 2. 切换到目标实例 Context

```cpp
aclshmemx_instance_ctx_set(2);

// 后续 SHMEM API 均作用于实例 2
void *ptr = aclshmem_malloc(1024);
aclshmem_free(ptr);
```

### 3. 获取当前实例

```cpp
aclshmem_instance_ctx *ctx = aclshmemx_instance_ctx_get();
// 当前主要用于读取 id，其他能力见上一节
uint64_t cur_id = (ctx == nullptr) ? 0 : ctx->id;
```

### 4. 释放实例

```cpp
aclshmem_finalize(2);
```
也可直接参照examples/multi_instance目录

## 规格与限制（按代码实现）

| 项目 | 规格 |
|---|---|
| `instance_id` 类型 | `uint64_t` |
| 默认实例 | `instance_id = 0` |
| PE 上限 | `ACLSHMEM_MAX_PES = 16384` |
| 本地内存上限 | `ACLSHMEM_MAX_LOCAL_SIZE = 40GB` |
| 非 0 实例 Team 行为 | 初始化时仅保留 world team 槽位；team split/translate/get_config 等返回 `ACLSHMEM_NOT_SUPPORTED` |
| `ip_port` 最大长度 | `ACLSHMEM_MAX_IP_PORT_LEN = 64`（含结尾 `\0`） |
| default 多实例端口范围 | 依赖 `SHMEM_INSTANCE_PORT_RANGE=start:end` |
| default 端口分配规则 | 输入端口必须是 `0`；实际端口 = `start_port + instance_id` |
| default 可用实例 ID 范围 | `instance_id <= end_port - start_port`（含边界） |

## 环境变量与运行要求

### Kernel侧使用要求
- 编译时必须类似 `add_compile_definitions(MULTI_INSTANCE)` 添加 `MULTI_INSTANCE` 编译宏
- 多实例使用时必须在算子开头使用 `aclshmemx_instance_ctx_set(id)` 来指定当前实例
- 算子执行时可以做实例间切换

### default 多实例模式
- 必须设置 `SHMEM_INSTANCE_PORT_RANGE`，格式 `start:end`
- 初始化属性中的 `ip_port` 端口必须为 `0`，例如 `(attr.ip_port = "tcp://x.x.x.x:0")`

### unique id 模式
- 仍遵循现有 `SHMEM_UID_SESSION_ID` / `SHMEM_UID_SOCK_IFNAME` 规则

## 注意事项

- 重复创建同一 `instance_id` 时：会有WARN日志提示并立即返回（不会重复创建）。
- `aclshmem_finalize(instance_id)` 会先切换到目标实例，再执行释放，避免误释放。
- 实例上下文相关接口路径由互斥锁保护。

## 与单实例模式对比

| 特性 | 单实例模式 | 多实例模式 |
|---|---|---|
| 初始化标识 | 默认 `instance_id=0` | `instance_id>0` |
| 资源隔离 | 全局共享 | 按实例隔离 |
| 上下文切换 | 不需要 | 需要 `aclshmemx_instance_ctx_set` |
| 生命周期管理 | `aclshmem_finalize()` | `aclshmem_finalize(instance_id)` |
| Team 高级操作 | 支持 | 仅 `instance_id=0` 支持 |

## 相关代码与文档

- Host 初始化与多实例实现：`src/host/init/shmem_init.cpp`
- Device 多实例状态路由：`src/device/shmemi_device_common.hpp`
- Team 多实例限制逻辑：`src/host/team/shmem_team.cpp`
- 对应示例：`examples/multi_instance`

---

最后更新：2026-03-25
