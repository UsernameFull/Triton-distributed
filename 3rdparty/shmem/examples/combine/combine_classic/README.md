# 经典 MoE Combine 示例

本示例实现非量化 MoE 的经典 combine 算子，对应设计文档 `DOC/moe_dispatch_combine_non_quant_architecture.md`。Combine 消费 dispatch 阶段生成的 expert 输出和辅助元数据，将每个 topK expert 的结果回传到原始 token 所在 PE，并完成加权归约。

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

## 实现逻辑

Kernel 启动 `pe_size` 个 AIV core。发送回传阶段中，`core_id == src_rank` 的 core 负责将本 PE 上的 expert 输出写回对应来源 rank。

主要流程：

1. 根据 `ep_recv_count` 遍历每个 `(local_expert_id, src_rank_id)` segment。
2. 从 `assist_info_for_combine` 取出原始 `token_id` 和 `topk_id`。
3. 使用 MTE `put_nbi` 将 expert 输出写回来源 PE 的 combine window。
4. 写入 status ready flag，通知来源 PE 对应 topK 结果已可读。
5. 本 PE 等待本地每个 token 的全部 topK status ready。
6. 按 `expert_scales` 做加权求和，写出 `x_out`。
7. 清理 status 并做 core 间同步。

当前经典版本数据面和控制面都使用 MTE。

## 默认 shape

```text
H = 7168
TopK = 8
BS、PEs、expertPerPe 可配置
```

可以通过 `-h`、`-topk` 或性能 sweep 参数覆盖默认值。

## 构建

在仓库根目录执行：

```bash
bash scripts/build.sh -examples
```

## 运行

基础 2 卡测试：

```bash
cd examples/combine/combine_classic
bash scripts/run.sh -pes 2 -bs 8 -expertPerPe 2 -type int32_t
```

8 卡、64 expert 测试：

```bash
cd examples/combine/combine_classic
bash scripts/run.sh -pes 8 -bs 8 -expertPerPe 8 -type int32_t
```

float16 正确性测试：

```bash
bash scripts/run.sh -pes 2 -bs 16 -expertPerPe 2 -type float16_t
```

覆盖 TopK：

```bash
bash scripts/run.sh -pes 2 -bs 8 -topk 4 -expertPerPe 2 -type int32_t
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

`bfloat16_t` 当前未在 combine 示例中实例化。原因是 CANN 9.0 beta 后端不支持 combine 累加路径需要的标量 bf16 cast，脚本会主动拒绝 `-type bfloat16_t`。

## 测试流程

一次功能测试会执行：

1. 删除旧的 `golden/` 和 `output/`。
2. 使用父目录公共脚本 `../scripts/data_gen.py` 生成输入、dispatch 元数据和 golden。
3. 启动每个 PE 的 `combine` 进程。
4. 写出 `output/x_out_<rank>.bin`。
5. 使用父目录公共脚本 `../scripts/check_combine.py` 比较所有 rank 输出。

## 性能测试

单 shape profiling：

```bash
cd examples/combine/combine_classic
bash scripts/run.sh --perf -pes 2 -bs 8 -expertPerPe 2 -type int32_t \
    --warmup 5 --loops 50
```

BS sweep：

```bash
bash scripts/run.sh --perf -pes 2 -type int32_t --bs-list 8,16,32 \
    --warmup 5 --loops 50
```

多卡、多 shape sweep：

```bash
bash scripts/run.sh --perf --pes-list 2,4,8 --bs-list 8,16 \
    --expert-per-pe-list 2,8 -type int32_t --prof-pe all
```

CSV 指标包括：

- `full_op`：完整 combine，包括回传通信、status wait、加权归约、清理和同步。
- `comm_only`：Stage 1 回传通信及必要的完成/status 协议。

CSV 文件写入 `output/perf/`：

```text
combine_perf_rank0.csv
combine_perf_rank1.csv
combine_perf_summary.csv  # 使用 --prof-pe all 时生成
```

前六列兼容 `examples/utils/perf_data_process.py`：

```text
DataSize/B,Npus,Blocks,UBsize/KB,Bandwidth/GB/s,CoreMaxTime/us
```

附加列包括 `Metric`、`GlobalDataSize/B`、`PerPeBandwidth/GB/s`、`BS`、`H`、`TopK`、`ExpertPerPe`、`Dtype`、`Warmup`、`Loops`、`ProfPe` 和 `CaseId`。

可使用 `--analyse plot` 或 `--analyse md` 在性能测试后调用统一性能报告脚本。
