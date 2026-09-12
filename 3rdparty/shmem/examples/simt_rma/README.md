## 样例介绍

本样例旨在展示 SIMD 与 SIMT 混合编译模式下，SIMT 远程内存访问（RMA）接口的典型使用方法。该类接口主要包含以下三种形式：

1. `__simt_callee__ inline void aclshmem_{NAME}_{op}(__gm__ TYPE *dst, __gm__ TYPE *src, uint32_t elem_size, int32_t pe)`
2. `__simt_callee__ inline void aclshmem_{op}{BITS}(__gm__ void *dst, __gm__ void *src, size_t nelems, int32_t pe)`
3. `__simt_callee__ inline void aclshmem_{op}mem(__gm__ void *dst, __gm__ void *src, uint32_t elem_size, int32_t pe)`

上述接口名称中的占位符 `{}` 可选值如下表所示：

| 占位符 | 可选值 |
| --- | --- |
| `{op}` | `put`, `get` |
| `{NAME}` | `half`, `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `char`, `bfloat16` |
| `{BITS}` | `8`, `16`, `32`, `64`, `128` |

这三种接口的核心功能均为实现连续内存区域的数据传输，其区别在于数据长度的指定方式：
- **第一种接口**：基于每个传输元素的具体数据类型（如 `half`、`float` 等）进行描述。
- **第二种接口**：基于每个传输元素的比特位大小（如 `8`、`16` 等）进行描述。
- **第三种接口**：直接指定需要传输的总内存字节大小。

### 样例执行流程

本样例通过以下流程演示 RMA 接口的具体工作机制：

1. **环境初始化**：每个计算单元（PE）初始化 3 块大小相同的对称内存。其中，第一块内存的数据初始化为 `[my_pe + 0, ..., my_pe + size - 1]`，第二块和第三块内存的数据初始化为 `-1`。
2. **GET 操作演示**：每个 PE 均调用 `get` 接口，将逻辑上属于**上一个 PE** 的第一块内存中的数据，拉取并写入至自身的第二块内存中。
3. **PUT 操作演示**：每个 PE 均调用 `put` 接口，将自身第一块内存中的数据，推送并写入至逻辑上属于**下一个 PE** 的第三块内存中。
4. **结果校验**：通信操作完成后，各 PE 将自动比对内存中的数据，验证数据传输的正确性。

## 支持的设备

- Ascend950

## 使用方式

1. **编译项目**  
   在 `shmem/` 根目录下执行编译脚本：
   ```bash
   bash scripts/build.sh -examples -enable_simt -soc_type Ascend950
   ```

2. **运行simt_rma示例程序**  
   进入示例目录并执行运行脚本：
   ```bash
   cd examples/simt_rma
   bash scripts/run.sh
   ```