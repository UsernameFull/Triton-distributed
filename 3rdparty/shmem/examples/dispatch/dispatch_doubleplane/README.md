# 双平面 MoE Dispatch 示例

本示例实现非量化 MoE dispatch 的双平面版本。它保持与 `examples/dispatch/dispatch_classic` 相同的外部语义和输出格式，但在 payload 传输上同时启用 MTE 与 SDMA，根据 segment 大小自适应选择传输路径。

双平面路径依赖 SHMEM SDMA 能力。SDMA 功能要求 CANN 9.0.0 及以上版本的尝鲜版环境，并需要安装匹配硬件平台的 toolkit 和 ops-legacy 软件包。基础安装和独立 SDMA demo 可参考 `examples/sdma/README.md`。

## 为什么使用双平面

经典 dispatch 中所有 token payload 都走 MTE。对于负载均衡或小 shape，这种方式简单直接；但 MoE 路由天然可能倾斜，某些 `(dst_rank, dst_local_expert)` segment 会集中大量 token，形成远端大块连续数据传输。

双平面的目标是：

- 小段继续走 MTE，避免 SDMA 提交和同步开销。
- 大段走 SDMA，利用 SDMA 更适合大批量远端数据搬运的特性。
- 控制面仍走 MTE，保证 ready/count/assist 协议简单可靠。
- 对外保持经典 dispatch 的输出一致，便于替换和对比性能。

适合优先尝试双平面的场景：

- `bs`、`h`、`topk` 较大，远端 payload 总量较高。
- expert 路由分布不均，存在明显大 segment。
- 希望比较 MTE-only 与 MTE+SDMA 双通路在通信阶段的收益。

如果 shape 很小或路由非常均匀，双平面可能不会明显优于经典版，因为 SDMA 路径本身有 issue、event 和 quiet 的固定成本。

## 方案设计

Host 初始化 SHMEM 时同时启用：

```text
ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_SDMA
```

Kernel 仍然启动 `pe_size` 个 AIV core，每个 active core 负责一个目标 rank。对每个 `(dst_rank, dst_local_expert)` segment，先统计该 segment 的 token 数，再判断是否启用 SDMA。

SDMA 判定逻辑可以理解为：

```text
segment_bytes = token_count * h * sizeof(T)

use_sdma =
    dst_rank != my_rank &&
    token_count > 0 &&
    segment_bytes >= 2MB &&
    segment_bytes > 当前 PE 的远端平均 segment 字节数
```

代码中为了避免整数除法截断，使用交叉相乘：

```text
segment_bytes * threshold_den > threshold_num
```

其中：

```text
threshold_num = remote_tokens * h * sizeof(T)
threshold_den = max(pe_size - 1, 1) * local_expert_num
```

也就是说，只有“大于 2MB 且大于当前 PE 远端平均段大小”的远端大段才走 SDMA。本地段、空段、小段和普通段仍走 MTE。

## 实现逻辑

主要阶段：

1. 每个 active core 统计发往目标 rank 的各 local expert segment 的 `segment_counts`。
2. 根据远端总 token 数计算平均阈值，并为每个 segment 标记 `sdma_flags`。
3. 先提交所有 SDMA payload：使用 `aclshmemx_sdma_put_nbi` 非阻塞写远端 payload。
4. 每提交 256 次 SDMA issue 后调用 `aclshmemx_sdma_quiet`，避免 outstanding 请求无限积压。
5. SDMA payload 阶段结束后，如果仍有未 quiet 的请求，再统一 `sdma_quiet`。
6. 小段 payload 使用 MTE 直连路径传输。
7. 控制面信号统一用 MTE 写入，包括 `assist_info_for_combine`、ready flag 和 count signal。
8. 接收端等待 count/ready 后，按经典 dispatch 的顺序构造最终 `expand_x`、`assist_info_for_combine`、`ep_recv_count` 和 `expert_token_nums`。

注意：SDMA 只负责大段 payload 数据面；控制面仍然由 MTE 负责。这样可以避免对端看到 ready 但 payload 仍未完成的时序问题。

## 与经典 Dispatch 的关系

双平面的输入输出与经典 dispatch 一致：

- 固定容量对称 SHMEM data window。
- token 级 ready flag。
- 输出顺序为 `(local_expert_id, src_rank_id)`。
- `assist_info_for_combine = [src_rank_id, src_token_id, src_topk_id]`。

因此同一组 golden/check 脚本可以验证两条路径。功能正确性应与经典 dispatch 完全一致，差异主要体现在通信阶段性能。

## 构建

在仓库根目录执行：

```bash
bash scripts/build.sh -examples
```

## 运行

基础 2 卡测试：

```bash
cd examples/dispatch/dispatch_doubleplane
bash scripts/run.sh -pes 2 -bs 8 -h 16 -topk 2 -expertPerPe 2 -type int32_t
```

8 卡、64 expert 测试：

```bash
cd examples/dispatch/dispatch_doubleplane
bash scripts/run.sh -pes 8 -bs 8 -h 16 -topk 2 -expertPerPe 8 -type int32_t
```

脚本会自动生成输入、路由矩阵和 golden 数据，启动每个 PE 对应的进程，输出写入 `output/`，并校验经典 dispatch 的四个输出。

## 常用参数

```text
-pes <n>            PE/NPU 数量，单机示例要求 -gnpus 与 -pes 相同。
-bs <n>             每个 PE 的 token 数。
-h <n>              token hidden size。
-topk <n>           每个 token 路由的 expert 数。
-expertPerPe <n>    每个 PE 上的 local expert 数，范围为 [1, 1024]。
-type <dtype>       数据类型，支持 int32_t、float16_t、bfloat16_t。
-fnpu <id>          起始 NPU id，默认 0。
-ipport <url>       SHMEM bootstrap 地址，默认 tcp://127.0.0.1:8766。
```

`expertPerPe` 上限为 1024。Kernel 在 AI core 栈上为每个目标 local expert 分配固定工作区，超过该上限会被 host 侧拒绝。

## 如何选择使用

建议先用经典 dispatch 建立正确性基线，再用双平面对相同 shape 做性能对比。

推荐对比命令：

```bash
cd examples/dispatch/dispatch_classic
bash scripts/run.sh --perf -pes 8 -bs 32 -h 1024 -topk 2 -expertPerPe 8 -type int32_t --warmup 5 --loops 50

cd ../dispatch_doubleplane
bash scripts/run.sh --perf -pes 8 -bs 32 -h 1024 -topk 2 -expertPerPe 8 -type int32_t --warmup 5 --loops 50
```

如果 `comm_only` 降低，说明大段 payload 走 SDMA 对通信阶段有效；如果 `full_op` 收益不明显，需要结合 shape、路由倾斜度和后续 compact/同步成本判断。

## 性能测试

单 shape profiling：

```bash
cd examples/dispatch/dispatch_doubleplane
bash scripts/run.sh --perf -pes 2 -bs 8 -h 256 -topk 2 -expertPerPe 2 -type int32_t \
    --warmup 5 --loops 50
```

多 shape、多卡数 sweep：

```bash
cd examples/dispatch/dispatch_doubleplane
bash scripts/run.sh --perf --pes-list 2,4,8 --bs-list 8,16,32 --h-list 64,256,1024 \
    --topk-list 2 --expert-per-pe-list 2,8 -type int32_t --prof-pe all \
    --warmup 5 --loops 50
```

CSV 指标包括：

- `full_op`：完整 dispatch，包括通信、元数据构造、compact 和同步。
- `comm_only`：Stage 1 payload 通信及必要的元数据/status 协议。

单 rank 文件名为 `dispatch_doubleplane_perf_rank<rank>.csv`。使用 `--prof-pe all` 时，脚本会轮流 profile 每个 PE，并生成 `dispatch_doubleplane_perf_summary.csv`。
