# MoE Combine 示例目录

本目录聚合 combine 相关示例，公共数据生成和校验脚本放在父级 `scripts/`，具体算子实现放在独立子目录。

## 目录结构

```text
examples/combine/
├── scripts/
│   ├── data_gen.py          # classic 与 doubleplane 共用的数据生成脚本
│   └── check_combine.py     # classic 与 doubleplane 共用的结果校验脚本
├── combine_classic/
│   ├── main.cpp
│   ├── combine_kernel.cpp
│   └── scripts/run.sh
└── combine_doubleplane/
    ├── main.cpp
    ├── combine_doubleplane_kernel.cpp
    └── scripts/run.sh
```

## 如何使用

经典 combine：

```bash
cd examples/combine/combine_classic
bash scripts/run.sh -pes 2 -bs 8 -expertPerPe 2 -type int32_t
```

双平面 combine：

```bash
cd examples/combine/combine_doubleplane
bash scripts/run.sh -pes 2 -bs 8 -h 16 -topk 2 -expertPerPe 2 -type int32_t
```

每个 `run.sh` 都会在当前 case 目录下生成 `golden/` 和 `output/`，并调用父级公共脚本 `../scripts/data_gen.py`、`../scripts/check_combine.py`。

## 选择建议

先使用 `combine_classic` 建立正确性和性能基线；当 shape 较大、回传 payload 存在明显大 segment 时，再使用 `combine_doubleplane` 对比 `comm_only` 和 `full_op` 指标。
