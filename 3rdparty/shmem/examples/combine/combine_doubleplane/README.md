# 双平面 MoE Combine 示例

本示例实现非量化 MoE combine 的双平面版本。它保持与 `examples/combine/combine_classic` 相同的输入输出和校验语义，但在 expert 输出回传阶段同时启用 MTE 与 SDMA，根据 segment 大小自适应选择传输路径。

双平面路径依赖 SHMEM SDMA 能力。SDMA 功能要求 CANN 9.0.0 及以上版本的尝鲜版环境，并需要安装匹配硬件平台的 toolkit 和 ops-legacy 软件包。基础安装和独立 SDMA demo 可参考 `examples/sdma/README.md`。

## 为什么使用双平面

经典 combine 中所有 expert 输出回传都走 MTE。对于小 shape 或均匀路由，MTE-only 的固定开销较低；但当某些来源 rank 对应的 expert 输出较多时，回传 payload 会形成大批量远端写。

双平面的目标是：

- 小段继续走 MTE，避免 SDMA 固定开销。
- 大段走 SDMA，提高大 payload 远端写效率。
- status ready 等控制信号仍走 MTE，保证同步协议简单可靠。
- 输出 `x_out` 与经典 combine 完全一致，便于功能校验和性能对比。

适合优先尝试双平面的场景：

- `bs`、`h`、`topk` 较大，combine 回传总数据量较高。
- dispatch 后的 `ep_recv_count` 显示部分 `(local_expert_id, src_rank_id)` segment 明显偏大。
- 关注通信阶段 `comm_only` 性能，希望比较 MTE-only 与 MTE+SDMA 的差异。

如果每个 segment 都较小，SDMA 的 issue/event/quiet 成本可能抵消收益。

## 功能说明

输入包括 dispatch 输出：

- `expand_x`：dispatch 后按 expert 聚合的 token 数据。本示例的数据生成脚本使用 identity expert 计算，因此 `expand_x` 可直接视为 expert 输出。
- `assist_info_for_combine`：每条 expert 输出对应的来源 `[src_rank_id, src_token_id, src_topk_id]`。
- `ep_recv_count`：dispatch 阶段生成的 segment 累计接收计数。

还需要 combine 自身输入：

- `expert_ids`：每个 token/topK 对应的 expert id。
- `expert_scales`：每个 token/topK 的加权系数。

输出为：

- `x_out`：每个 token 的最终 combine 结果。

校验公式：

```text
x_out[token] = sum(topk_output[token, topk] * expert_scales[token, topk])
```

## 方案设计

Host 初始化 SHMEM 时同时启用：

```text
ACLSHMEM_DATA_OP_MTE | ACLSHMEM_DATA_OP_SDMA
```

Kernel 仍然启动 `pe_size` 个 AIV core，每个 active core 负责一个来源 rank。对每个 `(src_rank, local_expert)` segment，先通过 `ep_recv_count` 得到该 segment 的 token 数，再判断是否启用 SDMA。

SDMA 判定逻辑可以理解为：

```text
segment_bytes = token_count * h * sizeof(T)

use_sdma =
    src_rank != my_rank &&
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

1. 每个 active core 根据 `ep_recv_count` 遍历自己负责来源 rank 的所有 local expert segment。
2. 计算当前 PE 的远端平均 segment 字节数，并对每个 segment 判断 `use_sdma`。
3. 对大段先提交 SDMA payload：使用 `aclshmemx_sdma_put_nbi` 非阻塞写回来源 PE 的 combine window。
4. 每提交 256 次 SDMA issue 后调用 `aclshmemx_sdma_quiet`，避免 outstanding 请求无限积压。
5. 对小段使用 MTE 直连路径写 payload，并立即写 status ready。
6. SDMA payload 阶段完成后统一 `sdma_quiet`，再用 MTE 为 SDMA 段写 status ready。
7. 来源 PE 等待本地每个 token 的全部 topK status ready。
8. 按 `expert_scales` 做加权求和，写出 `x_out`，再清理 status 并同步退出。

注意：SDMA 只负责大段 payload 数据面；status ready 控制面仍由 MTE 写入。这样可以保证 status 不会早于 payload 可见。

## 与经典 Combine 的关系

双平面的输入输出与经典 combine 一致：

- 消费 `expand_x`、`assist_info_for_combine`、`ep_recv_count`。
- 使用相同的 `expert_ids` 和 `expert_scales`。
- 产生相同语义的 `x_out`。
- 使用相同的 golden/check 脚本验证正确性。

因此可以先用经典 combine 建立正确性基线，再用双平面对相同 shape 做性能对比。

## 构建

在仓库根目录执行：

```bash
bash scripts/build.sh -examples
```

## 运行

基础 2 卡测试：

```bash
cd examples/combine/combine_doubleplane
bash scripts/run.sh -pes 2 -bs 8 -h 16 -topk 2 -expertPerPe 2 -type int32_t
```

8 卡、64 expert 测试：

```bash
cd examples/combine/combine_doubleplane
bash scripts/run.sh -pes 8 -bs 8 -h 16 -topk 2 -expertPerPe 8 -type int32_t
```

脚本会自动生成 combine 输入和 golden 输出，启动每个 PE 对应的进程，输出写入 `output/x_out_<rank>.bin`，并执行父目录公共脚本 `../scripts/check_combine.py` 校验结果。

## 常用参数

```text
-pes <n>            PE/NPU 数量，单机示例要求 -gnpus 与 -pes 相同。
-bs <n>             每个 PE 的 token 数。
-h <n>              token hidden size，默认 7168。
-topk <n>           每个 token 路由的 expert 数，默认 8。
-expertPerPe <n>    每个 PE 上的 local expert 数。
-type <dtype>       数据类型，支持 int32_t、float16_t。
-fnpu <id>          起始 NPU id，默认 0。
-ipport <url>       SHMEM bootstrap 地址，默认 tcp://127.0.0.1:8767。
```

`bfloat16_t` 当前未在 combine doubleplane 示例中实例化。原因是 CANN 9.0 beta 后端不支持 combine 累加路径需要的标量 bf16 cast，脚本会主动拒绝 `-type bfloat16_t`。

## 如何选择使用

推荐用同一组 shape 对比经典版和双平面：

```bash
cd examples/combine/combine_classic
bash scripts/run.sh --perf -pes 8 -bs 32 -h 7168 -topk 8 -expertPerPe 8 -type int32_t --warmup 5 --loops 50

cd ../combine_doubleplane
bash scripts/run.sh --perf -pes 8 -bs 32 -h 7168 -topk 8 -expertPerPe 8 -type int32_t --warmup 5 --loops 50
```

优先观察 `comm_only`。如果 `comm_only` 降低，说明大段回传走 SDMA 对通信阶段有效；如果 `full_op` 收益较小，需要结合 status wait、加权归约和同步开销一起分析。

## 性能测试

单 shape profiling：

```bash
cd examples/combine/combine_doubleplane
bash scripts/run.sh --perf -pes 2 -bs 8 -h 256 -topk 2 -expertPerPe 2 -type int32_t \
    --warmup 5 --loops 50
```

多 shape、多卡数 sweep：

```bash
cd examples/combine/combine_doubleplane
bash scripts/run.sh --perf --pes-list 2,4,8 --bs-list 8,16,32 --h-list 64,256,1024 \
    --topk-list 2 --expert-per-pe-list 2,8 -type int32_t --prof-pe all \
    --warmup 5 --loops 50
```

CSV 指标包括：

- `full_op`：完整 combine，包括回传通信、status wait、加权归约、清理和同步。
- `comm_only`：Stage 1 回传通信及必要的完成/status 协议。

单 rank 文件名为 `combine_doubleplane_perf_rank<rank>.csv`。使用 `--prof-pe all` 时，脚本会轮流 profile 每个 PE，并生成 `combine_doubleplane_perf_summary.csv`。
