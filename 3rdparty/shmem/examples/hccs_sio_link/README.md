# HCCS SIO Link

HCCS/SIO 链路测试工具，用于验证 NPU 之间 SIO / HCCS 链路的正确性。

> **注意**：当前 CANN 版本暂未支持，预计 CANN 9.1.0 版本支持该功能。

## 链路说明

对于 A3 芯片，一个 NPU 包含两个 Die，两 Die 之间通过 SIO 链路互连。本方案在 SIO 链路基础上新增 HCCS 链路，实现 SIO/HCCS 双路并行传输，从而提高 Die 间数据传输速度。

- **SIO**：Die 间原有互连链路。shmem 初始化后默认获得的 VA 访问对端 Die 时即走 SIO 链路。

- **HCCS**：新增的 Die 间互连链路。通过 [`aclrtMemMapSelectedLink`](https://gitcode.com/cann/runtime/blob/master/docs/03_api_ref/11-04_虚拟内存管理.md#aclrtmemmapselectedlink) 将新 VA 映射到与原 SIO VA 相同的 PA，并使用 `ACL_RT_MEM_LINK_IDX_1` 选中 HCCS 链路，此时通过新 VA 访问即走 HCCS 链路。

- **SIO + HCCS 双路并行**：同时使用 SIO 和 HCCS 两条链路传输数据，充分利用双链路带宽提升传输性能。

## 核心函数说明

### `setup_hccs_mapping`

建立 HCCS 链路映射，为当前 PE 创建一条通过 HCCS 链路访问对端 PE 堆内存的虚拟地址通道。该函数执行以下四个步骤：

1. **获取本地堆基址**：调用 `aclshmemx_get_heap_base()` 获取当前 PE 的对称堆基址。
2. **转换为对端地址**：调用 `aclshmem_ptr()` 将本地堆基址转换为对端 PE 的可访问虚拟地址（`peer_heap_base`），该地址走 SIO 链路。
3. **预留虚拟地址空间**：调用 `aclrtReserveMemAddress()` 在当前 PE 的虚拟地址空间中预留一段未映射的 VA 区域（`hccs_ptr`）。
4. **映射到 HCCS 链路**：调用 [`aclrtMemMapSelectedLink`](https://gitcode.com/cann/runtime/blob/master/docs/03_api_ref/11-04_虚拟内存管理.md#aclrtmemmapselectedlink) 将预留的 VA 映射到与 `peer_heap_base` 所对应的物理地址（PA），并指定 `ACL_RT_MEM_LINK_IDX_1` 选中 HCCS 链路。此时通过 `hccs_ptr` 访问即走 HCCS 链路，而通过原 SIO 路径的 VA 访问仍走 SIO 链路——两条 VA 指向同一 PA，但走不同物理链路。

> **关键原理**：SIO 和 HCCS 两条链路共享同一物理地址（PA），但通过不同的虚拟地址（VA）和链路索引实现分流，从而支持双路并行传输。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `peer` | `int` | 对端 PE 编号，即需要建立 HCCS 映射的目标 PE |
| `local_mem_size` | `uint64_t` | 与 `aclshmemx_init_attr` 初始化时设定的对称堆大小相同（字节），实际映射区域大小为 `local_mem_size + ACLSHMEM_EXTRA_SIZE` |
| `hccs_ptr` | `void **` | 输出参数，成功时返回 HCCS 链路映射的虚拟地址基址；失败时值取决于失败发生的步骤，若 `aclrtReserveMemAddress` 已成功但后续步骤失败，`*hccs_ptr` 可能包含已预留但未映射的地址，需通过 `teardown_hccs_mapping` 或 `HccsMappingGuard` 释放以防泄漏 |

#### 返回值

- `true`：HCCS 映射建立成功，`*hccs_ptr` 为有效的 HCCS 虚拟地址基址。
- `false`：映射过程中任一步骤失败，函数会打印错误信息并返回。

### `teardown_hccs_mapping`

与 `setup_hccs_mapping` 配对的清理函数，用于释放 HCCS 映射资源：

1. 调用 `aclrtUnmapMem()` 解除 VA 到 PA 的映射关系。
2. 调用 `aclrtReleaseMemAddress()` 释放预留的虚拟地址空间。

> **注意**：在示例代码中，`HccsMappingGuard` 结构体利用 RAII 机制在析构时自动调用 `teardown_hccs_mapping`，确保映射资源不会泄漏。

## 编译

本示例需要通过 `-examples` 编译选项启用，CMake 会自动检测当前 CANN 版本是否支持 `aclrtMemMapSelectedLink` 函数，若支持则自动编译本示例：

```bash
bash scripts/build.sh -examples
```

编译产物：`build/bin/hccs_sio_link`

## 前置条件

- 已按上述方式编译 shmem 项目
- 已设置 `ASCEND_HOME_PATH` 环境变量

> **多实例说明**：`aclshmemx_get_heap_base` 返回当前活跃实例的堆基址。在多实例场景下，需先通过 `aclshmemx_instance_ctx_set_impl` 切换到目标实例，再调用 `aclshmemx_get_heap_base`。

## 用法

本工具通过 `run.sh` 脚本启动，脚本会为每个 PE 启动一个后台进程，各 PE 之间通过 shmem 初始化建立通信。

### 运行命令

```bash
bash run.sh [选项]
```

### 典型用例

```bash
# 默认配置：2 个 PE，SIO+HCCS 全链路测试，4KB 数据，int 类型
bash run.sh

# 指定 4 个 PE，仅测 HCCS 链路
bash run.sh -pes 4 -mode hccs

# 指定 8 个 PE，8MB 数据，fp32 类型，仅测 SIO 链路
bash run.sh -pes 8 -size 8 -type fp32 -mode sio

# SIO + HCCS 混合测试（3/5 数据走 SIO，2/5 数据走 HCCS）
bash run.sh -mode mixed
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-ipport` | `tcp://127.0.0.1:8766` | 通信初始化地址 |
| `-pes` | `2` | 参与测试的 PE 总数（与 NPU 数量一致） |
| `-fpe` | `0` | 首个 PE 编号 |
| `-fnpu` | `0` | 首个 NPU 编号 |
| `-type` | `int` | 测试数据类型：`int` / `int64` / `fp32` |
| `-mode` | `all` | 测试模式（见下表） |
| `-size` | `4` | 每个 PE 的数据大小（KB） |

### 测试模式

| 模式 | 说明 |
|------|------|
| `sio` | SIO 链路正确性测试 |
| `hccs` | HCCS 链路正确性测试 |
| `all` | SIO + HCCS 全链路正确性测试 |
| `mixed` | SIO + HCCS 混合正确性测试（3/5 数据走 SIO，2/5 数据走 HCCS） |

## 输出示例

正确性测试：

```
PE 0: [SIO] path verification PASSED for PE 1
PE 1: [SIO] path verification PASSED for PE 0
PE 0: [HCCS] path verification PASSED for PE 1
PE 1: [HCCS] path verification PASSED for PE 0
```

混合测试：

```
PE 0: [MIXED-SIO] path verification PASSED for PE 1
PE 0: [MIXED-HCCS] path verification PASSED for PE 1
PE 1: [MIXED-SIO] path verification PASSED for PE 0
PE 1: [MIXED-HCCS] path verification PASSED for PE 0
```
