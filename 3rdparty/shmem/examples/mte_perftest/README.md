# mte_perftest

## 示例概述

mte_perftest是用于测试AscendC::DataCopy和shmem MTE（Memory Transfer Engine）性能的参数化测试示例集合，包含两个子示例：

- **ascendc_perftest**：测试AscendC::DataCopy性能
- **shmem_perftest**：测试shmem MTE性能

该示例可以帮助用户对比两种数据传输方式的性能表现。**该脚本测试结果仅做参考，性能以实际场景为准**

## Python依赖

如果需要生成性能图表和Markdown报告，需要安装以下Python依赖：

```bash
pip install pandas matplotlib seaborn numpy tabulate
```

## 使用说明

### 快速开始

在shmem根目录下编译并运行：

```bash
# 编译示例
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -examples

# 如需运行 --memory-type dram，需启用CANN模式编译
bash scripts/build.sh -examples -cann

# 运行两个示例（默认模式，仅生成CSV）
cd examples/mte_perftest
bash run.sh -m all -t put -d float -fpe 0

# 运行两个示例，并生成性能图表
bash run.sh -m all -t put -d float -fpe 0 -a plot

# 运行两个示例，并生成性能图表和Markdown报告
bash run.sh -m all -t put -d float -fpe 0 -a md
```

### 外层run.sh参数说明

| 参数 | 说明 | 默认值 |
|------|------|---------|
| `-t\|--test-type <type>` | 测试类型 (put\|get\|ub2gm_local\|ub2gm_remote\|gm2ub_local\|gm2ub_remote\|all) | put |
| `-d\|--datatype <type>` | 数据类型 (float\|int8\|int16\|int32\|int64\|uint8\|uint16\|uint32\|uint64\|char\|all) | float |
| `-b\|--block-size <size>` | 设置核数 | 32 |
| `--block-range <min> <max>` | 设置核数范围 | 32-32 |
| `-e\|--exponent <exponent>` | 设置数据量的幂数 | - |
| `--exponent-range <min> <max>` | 设置数据量的幂数范围 | 3-20 |
| `--loop-count <count>` | 设置循环次数 | 1000 |
| `--ub-size <size>` | 设置UB size(KB), 默认16 | 16 |
| `--memory-type <hbm\|dram>` | 设置shmem_perftest使用的SHMEM内存类型 | hbm |
| `-pes <size>` | 设置PE大小 | 2 |
| `-ipport <ip:port>` | 设置IP端口 | tcp://127.0.0.1:8764 |
| `-gnpus <num>` | 设置NPU数量 | 2 |
| `-fnpu <id>` | 设置首个NPU ID | 0 |
| `-fpe <id>` | 设置首个PE ID | 0 |
| `-m\|--mode <ascendc\|shmem\|all>` | 设置运行模式 (ascendc=只跑ascendc, shmem=只跑shmem, all=全跑), 默认all |
| `-a\|--analyse <none\|plot\|md>` | 设置分析模式 (none=不生成, plot=只生成图, md=同时生成图和md), 默认none |

### 运行模式

- **all（默认）**：同时运行ascendc_perftest和shmem_perftest
- **ascendc**：只运行ascendc_perftest
- **shmem**：只运行shmem_perftest

### DRAM内存测试约束

`--memory-type dram` 仅作用于 shmem_perftest，会使用 `aclshmemx_malloc(..., HOST_SIDE)` 分配Host侧DRAM内存。该功能依赖CANN模式，编译时必须使用：

```bash
bash scripts/build.sh -examples -cann
```

DRAM测试需要运行环境支持Host侧DRAM内存访问，相关硬件和可用内存约束可参考 [rma_d2h_demo](../rma_d2h_demo/README.md) 的“约束限制”章节。mte_perftest默认配置1GB本地内存；当测试参数需要更大本地内存时，程序会按数据量自动上调，运行前需确保可用DRAM空间大于实际本地内存配置。

## 输出结果

运行完成后，结果会统一放在 `examples/mte_perftest/output/` 目录下：

```
examples/mte_perftest/output/
├── ascendc_perftest/    # ascendc_perftest测试结果
│   ├── 0_put_float.csv
│   └── picture/          # （使用--plot时生成）性能图表
│       └── 0_put_float/
│           ├── 0_put_float_UBsize_compare.png
│           ├── 0_put_float_Core_compare.png
│           ├── 0_put_float_bandwidth_max_heatmap.png
│           └── 0_put_float_bandwidth_mean_heatmap.png
├── shmem_perftest/       # shmem_perftest测试结果
│   ├── put_float_0.csv
│   └── picture/          # （使用--plot时生成）性能图表
│       └── put_float_0/
│           ├── put_float_0_UBsize_compare.png
│           ├── put_float_0_Core_compare.png
│           ├── put_float_0_bandwidth_max_heatmap.png
│           └── put_float_0_bandwidth_mean_heatmap.png
└── performance_report.md  # （使用--markdown时生成）性能测试报告
```

## 单独测试子示例

如果需要单独测试某个子示例，请进入对应子目录查看详细README：

- **ascendc_perftest**：请参考 [ascendc_perftest/README.md](./ascendc_perftest/README.md)
- **shmem_perftest**：请参考 [shmem_perftest/README.md](./shmem_perftest/README.md)
