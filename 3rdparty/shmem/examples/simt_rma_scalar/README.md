## 样例介绍

本样例旨在展示 SIMD 与 SIMT 混合编译模式下，SIMT 远程内存访问（RMA）接口的典型使用方法。此样例为单个标量的传输。该类接口主要包含以下两种形式：

1. `__simt_callee__ inline void aclshmem_{NAME}_p(__gm__ TYPE *dst, const TYPE value, int pe)`
2. `__simt_callee__ inline void aclshmem_{NAME}_g(__gm__ TYPE *dst, const TYPE value, int pe)`

上述接口名称中的占位符 `{NAME}` 可选值如下表所示：

| 占位符 | 可选值 |
| --- | --- |
| `{NAME}` | `half`, `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `char`, `bfloat16` |

这两种接口的核心功能为实现单标量或小段数据传输：
- **`_p` 接口**：将指定的标量数值直接写入到指定计算单元（PE）的目标内存地址中。
- **`_g` 接口**：从指定计算单元（PE）的源内存地址中读取单个标量数值。

### 样例执行流程

本样例通过以下流程演示 RMA 单标量相关接口的具体工作机制：

1. **环境初始化**：首先获取当前计算单元的编号（`mype`）并计算出**下一个 PE** 的编号作为目标对端。
2. **PUT（写）操作演示**：当前 PE 调用写入接口（`aclshmem_int32_p`），将目标 PE 编号数值，作为标量直接写入至目标 PE 对应的对称内存中。
3. **GET（读）操作演示**：当前 PE 调用读取接口（`aclshmem_int32_g`），从目标 PE 的对称内存中，将存放的标量数值拉取回当前 PE，并存储在本地内存中。
4.  **结果校验**：通信操作完成后，各 PE 将自动比对内存中的数据，验证数据传输的正确性。


## 支持的设备

- Ascend950

## 使用方式

1. **编译项目**  
   在 `shmem/` 根目录下执行编译脚本：
   ```bash
   bash scripts/build.sh -examples -enable_simt -soc_type Ascend950
   ```

2. **运行 simt_rma_scalar 示例程序**  
   进入示例目录并执行运行脚本：
   ```bash
   cd examples/simt_rma_scalar
   bash scripts/run.sh
   ```