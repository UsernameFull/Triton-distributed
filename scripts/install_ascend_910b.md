# Ascend 910B 安装与测试

脚本从当前仓库安装 Ascend 后端、编译工具链和 SHMEM，然后执行数值与通信测试。
不升级 CANN 或驱动。完整测试集包含四卡子通信组测试，因此需至少 4 张可用 NPU；
仅检查基础依赖时需要 2 张。

## 使用

已有匹配 CANN 的 torch/torch_npu 时直接运行；本机避开报警的 0 号卡和已有任务的 3 号卡：

```bash
ASCEND_RT_VISIBLE_DEVICES=1,2,4,5 JOBS=8 \
bash scripts/install_ascend_910b.sh
```

可配置 `WORK_DIR=/data/triton-910b-build` 保存产物，默认是仓库下 `.ascend-910b/`。
`CANN_ENV` 指定 CANN 的 set_env.sh；`PYTHON` 指定基础 Python（默认 python3）。
虚拟环境使用 `--system-site-packages` 复用 torch/torch_npu，不是完全隔离环境。

只有 CANN、尚无 PyTorch 时，传入与 CANN/Python/主机架构匹配的 wheel：

```bash
TORCH_WHEEL=/path/to/torch.whl \
TORCH_NPU_WHEEL=/path/to/torch_npu.whl \
ASCEND_RT_VISIBLE_DEVICES=1,2,4,5 \
bash scripts/install_ascend_910b.sh
```

也可使用 `TORCH_VERSION` 和 `TORCH_NPU_VERSION` 指定精确版本，从当前 pip 源安装。
已有包默认检查并复用；显式指定版本或 wheel 则执行安装。不推测 CANN 版本兼容映射，
也不会在导入失败时盲目升级已有 torch。除版本检查外还实际执行两张卡的矩阵计算。

缺少系统编译工具时，root 下通过 apt-get 或 dnf 安装；普通用户会得到管理员安装提示。
优先使用 clang-15，否则接受 clang >=15 引导构建指定 LLVM。也可指定 CC、CXX、LD。
手动准备工具的示例：

```bash
# Debian / Ubuntu
sudo apt-get install -y git curl build-essential clang-15 lld-15 ccache python3-venv python3-dev
# openEuler
sudo dnf install -y git curl gcc gcc-c++ make clang lld ccache python3-devel
```

只检查、安装基础依赖并验证两卡计算：

```bash
ASCEND_RT_VISIBLE_DEVICES=1,2 \
bash scripts/install_ascend_910b.sh --prereqs-only
```

## 构建行为

需要联网访问 GitHub、GitCode 和 Python 包源，并预留充足的磁盘与编译内存。
`JOBS` 默认 4；`TEST_TIMEOUT` 默认 1800 秒，限制每个测试阶段，不限制源码构建总时长。

- LLVM 使用文档指定的 `fad3272286528b8a491085183434c5ad4b59ab92` 及其补丁。
  `LLVM_INSTALL_PREFIX` 可复用符合该提交及补丁要求的安装；LLVM_PATCH_URL 可指定补丁镜像。
- Triton 使用仓库锁定的 triton-ascend 和 AscendNPU-IR 头文件，并应用仓库内的两个补丁。
- 锁定的 IR 头文件版本没有 SHMEM 模板，运行时编译器单独构建固定提交
  `f16704a2545db71c75b1a4f9f8c325c3e236954e`，保留自己的 LLVM 依赖。
  NPUIR_RUNTIME_COMMIT 可覆盖提交，NPUIR_RUNTIME_BIN 可复用已构建的编译器 bin 目录。
- 首次生成 wheel 元数据之前建立 `python/triton` 的 Ascend 链接，确保 wheel 包含 Python 源文件。
  安装会切换这个仓库内的后端链接，并修改子模块中的补丁目标文件。
- 重跑允许已有的预期补丁，其他已跟踪修改会导致退出，避免覆盖已有工作。
  虚拟环境和成功安装的 LLVM 会复用；其他构建流程会再次执行。
- 不安装 FlashComm/CUDA/ROCm 运行环境；项目现有打包逻辑仍包含其他后端的编译资产。

## 测试与运行环境

按顺序验证：两卡张量计算、后端导入、HCCL AllReduce、整个 Ascend pytest 目录、
双卡 AllGather GEMM 数值校验。pytest 使用 importlib 导入模式，避免子进程加载未编译的源码包。
各阶段串行执行，上游 SHMEM 测试使用固定端口（包括 8666），不要并行启动多份脚本。
只有所有测试命令成功并正常退出，脚本才输出最终 PASS。

安装后复跑全部测试：

```bash
ASCEND_RT_VISIBLE_DEVICES=1,2,4,5 \
WORK_DIR=/data/triton-910b-build \
bash scripts/install_ascend_910b.sh --tests-only
```

加载运行环境：

```bash
source /data/triton-910b-build/env.sh
export ASCEND_RT_VISIBLE_DEVICES=1,2,4,5
```

日志保存在 `$WORK_DIR/logs/`，失败时输出阶段、退出码和日志路径。

## 本机验证配置

openEuler 24.03、Ascend 910B3、CANN 9.1.0，复用 clang 17.0.6、LLD 15.0.5、
torch 2.10.0+cpu、torch_npu 2.10.0.post4，虚拟环境安装 cmake 3.31.10。
此组合是本机验证结果，不是其他 CANN 版本的通用推荐。

本机产物：`/tmp/triton-dist-910b-validation/`，LLVM 复用 `/root/ascend-build/llvm-install`。
独立运行时编译器由本次源码构建，版本为 bishengir-compile 1.2.0。
基础环境的 vllm-ascend 版本约束已有冲突，本次未调整这些业务包。
0 号卡曾显示 Alarm，复跑时一个验证进程进入内核 D 状态，信号未能立即结束；本次验证避开该卡。

最终完整安装入口已正常退出，退出码为 0：两卡计算、后端识别、HCCL 全部通过，
Ascend 测试集 11/11 通过（149.07 秒），双卡 AllGather GEMM 两个 rank 均通过
`rtol=1e-3, atol=1e-3` 数值校验。

完整日志：`/tmp/triton-dist-910b-validation/logs/install-20261018T134131Z.log`。
使用现有工具链复跑本次完整入口的命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=1,2,4,5 \
WORK_DIR=/tmp/triton-dist-910b-validation \
LLVM_INSTALL_PREFIX=/root/ascend-build/llvm-install \
NPUIR_RUNTIME_BIN=/tmp/triton-dist-910b-validation/npuir-build/bin \
JOBS=48 TEST_TIMEOUT=1200 \
bash scripts/install_ascend_910b.sh
```

## 卸载后重新安装验证

2026-10-18 再次验证时，先卸载系统 clang、lld、ccache，以及系统和旧验证虚拟环境中的
shmem。保留 CANN、驱动、torch、torch_npu，使用全新的 `/tmp/triton-dist-910b-clean`
构建目录，移走原 Python/C++ build 目录。LLVM 和独立运行时编译器均重新构建，
未传入 LLVM_INSTALL_PREFIX 或 NPUIR_RUNTIME_BIN 复用旧工具链。

此次发现 GitHub Git 传输停滞，增加 LLVM_SOURCE_ARCHIVE 支持官方同提交源码包：

```bash
curl -fL --retry 3 \
  https://codeload.github.com/llvm/llvm-project/tar.gz/fad3272286528b8a491085183434c5ad4b59ab92 \
  -o /tmp/llvm-source.tar.gz
ASCEND_RT_VISIBLE_DEVICES=1,2,4,5 \
WORK_DIR=/data/triton-910b-clean \
LLVM_SOURCE_ARCHIVE=/tmp/llvm-source.tar.gz \
JOBS=8 bash scripts/install_ascend_910b.sh
```

源码包必须是该提交的官方 tar.gz，脚本核对顶层目录名及解包标记；自行提供的压缩包需保证可信。
本次下载文件的 SHA256：
`a45117e2bafdfbb791b46aa817c2b22d0dc1ffe708a7b4fc0acf7bff8c2fdc48`。

旧的系统工具安装分支还连带升级了 gcc、git、curl、系统 Python 3.11 及相关库；
当前基础 Python 3.12 和 torch 未替换。脚本已改为只把缺失包传给包管理器，
减少无关升级；安装新包所需的依赖仍由包管理器解析。

本次卸载后重装完整入口退出码为 0：HCCL 通过，Ascend 测试集 11/11 通过
（217.15 秒），AllGather GEMM 两个 rank 的数值校验通过。
完整日志：`/tmp/triton-dist-910b-clean/logs/install-20261018T161724Z.log`。
运行环境：`source /tmp/triton-dist-910b-clean/env.sh`。
修改后的“只安装缺失系统包”分支另外通过实际卸载并自动重装 ccache 验证，
此次仅安装 ccache，没有升级其他包。日志位于 `/tmp/triton-910b-clean-audit/system-recheck.log`。
本次脚本与说明更新尚未提交或推送 GitHub。
