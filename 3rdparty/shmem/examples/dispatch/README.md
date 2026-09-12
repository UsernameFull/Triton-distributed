# MoE Dispatch 示例目录

本目录聚合 dispatch 相关示例，公共数据生成和校验脚本放在父级 `scripts/`，具体算子实现放在独立子目录。

## 目录结构

```text
examples/dispatch/
├── scripts/
│   ├── data_gen.py           # classic 与 doubleplane 共用的数据生成脚本
│   └── check_dispatch.py     # classic 与 doubleplane 共用的结果校验脚本
├── dispatch_classic/
│   ├── main.cpp
│   ├── dispatch_kernel.cpp
│   └── scripts/run.sh
└── dispatch_doubleplane/
    ├── main.cpp
    ├── dispatch_doubleplane_kernel.cpp
    └── scripts/run.sh
```

## 如何使用

经典 dispatch：

```bash
cd examples/dispatch/dispatch_classic
bash scripts/run.sh -pes 2 -bs 8 -h 16 -topk 2 -expertPerPe 2 -type int32_t
```

双平面 dispatch：

```bash
cd examples/dispatch/dispatch_doubleplane
bash scripts/run.sh -pes 2 -bs 8 -h 16 -topk 2 -expertPerPe 2 -type int32_t
```

每个 `run.sh` 都会在当前 case 目录下生成 `golden/` 和 `output/`，并调用父级公共脚本 `../scripts/data_gen.py`、`../scripts/check_dispatch.py`。

## 选择建议

先使用 `dispatch_classic` 建立正确性和性能基线；当 `bs`、`h`、`topk` 较大且路由分布存在大 segment 时，再使用 `dispatch_doubleplane` 对比 `comm_only` 和 `full_op` 指标。
