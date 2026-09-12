### 使用方式

#### 1. 编译项目

在 `shmem/` 根目录下执行编译脚本：

```bash
bash scripts/build.sh -examples
```

#### 2. 运行 Dynamic-Tiling 示例程序

进入示例目录并执行运行脚本：

```bash
cd examples/dynamic_tiling
bash scripts/run.sh [comm_type] [data_type] [test_start_line] [test_collect_rows] [device_list]
```

##### 参数说明

| 参数 | 说明 | 取值示例 |
|------|------|---------|
| `comm_type` | 通信-计算融合算子类型 | `0`: MATMUL_ALLREDUCE<br>`1`: ALLGATHER_MATMUL<br>`2`: MATMUL_REDUCE_SCATTER |
| `data_type` | 数据类型 | `1`: FP16<br>`27`: BF16 |
| `test_start_line`（可选） | 测试起始行索引（对应`test_shapes.csv`中的行号，从0开始）<br>需与 `test_collect_rows` 一同指定，用于性能测试 | `0`, `10`, `...` |
| `test_collect_rows`（可选） | 每次采集性能数据的测试用例数量 | `5`, `10`, `...` |
| `device_list` | 指定运行的设备（NPU）编号列表，以逗号分隔 | `0,1`, `4,5,6,7` |

> 📌 **注意**：  
> - `peSize`由`device_list`中设备数量自动确定
> - 精度测试默认按顺序执行test_shapes.csv中定义的所有shape
> - 性能测试需指定test_start_line和test_collect_rows参数：从第test_start_line个shape开始，每次采集test_collect_rows个测试用例，持续执行直至文件末尾

##### 示例

- **精度测试示例**：  
  使用 NPU 0 和 1，运行 **MatMul-AllReduce** 精度测试，数据类型为FP16，`peSize = 2`：
  ```bash
  bash scripts/run.sh 0 1 0,1
  ```

- **性能测试示例**：  
  使用 NPU 4、5、6、7，运行 **AllGather-MatMul** 性能测试，数据类型为 BF16，从 `test_shapes.csv` 第0行开始，每 10 个 shape 采集一次 `msprof` 性能数据，`peSize = 4`：
  ```bash
  bash scripts/run.sh 1 27 0 10 4,5,6,7
  ```

#### 3. 配置计算规模

矩阵计算参数（包括 `M`, `K`, `N`, `Transpose A`, `Transpose B`）在配置文件中定义：

```
scripts/test_shapes.csv
```

请根据测试需求修改该文件，添加或调整测试用例的输入维度和属性。

---

✅ **提示**：  
- 确保设备编号正确且可用。  
- 建议在性能测试前清理无关进程，以保证数据准确性。  
- 性能数据默认输出至 `output/` 目录。
