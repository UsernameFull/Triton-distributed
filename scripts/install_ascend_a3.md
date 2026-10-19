# Ascend A3 编译安装

A3 入口复用 [A2/910B 安装流程](install_ascend_910b.md)，默认使用独立目录 `.ascend-a3`。
仓库 triton-ascend 提供 A2/A3 共用编译路径，SHMEM 文档也列出 A3 支持。
本脚本尚未在 A3 实机完成编译和通信验证；此前 910B 的测试结果不代表 A3 已通过。

先安装适配 A3 的驱动、CANN，以及匹配的 Python/torch/torch_npu，然后运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
WORK_DIR=/data/triton-a3 JOBS=8 \
bash scripts/install_ascend_a3.sh
```

脚本检测实际设备，不设置固定的 TRITON_ASCEND_ARCH；设备需识别为 Ascend910_93 系列
或 Ascend910C，Triton 编译目标需识别为 Ascend910_93 系列。不要用 A2 环境变量伪装目标。
完整测试要求至少四张可见逻辑 NPU；设备编号按目标主机实际情况选择。
不默认启用跨节点 RDMA；默认测试为单节点通信和 GEMM。

仅检查依赖和双卡计算：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
WORK_DIR=/data/triton-a3 bash scripts/install_ascend_a3.sh --prereqs-only
```

安装成功后重跑测试：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
WORK_DIR=/data/triton-a3 bash scripts/install_ascend_a3.sh --tests-only
```

运行环境为 `/data/triton-a3/env.sh`，日志在该目录的 `logs/` 下。
依赖缺失时的安装、指定 torch wheel、LLVM 源码包和工具链复用参数均与 A2 脚本相同。
不能据 A2 测试记录直接选择 A3 的 CANN/torch_npu 版本；应按目标机软件兼容要求准备。
A2/A3 入口会切换同一个仓库的后端链接并使用同一个 python/build，不要在同一 checkout 并发构建。
