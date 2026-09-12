1. **编译项目**  
   在 `shmem/` 根目录下执行编译脚本：
   ```bash
   bash scripts/build.sh -examples
   ```

2. **数据生成**  
   2.1 - **config 文件参数**:
   - `dataType` 数据类型，默认为2（INT8INT8_INT32_FP16），具体可见gen_data.py中的定义。
   - `peSize` 卡数，一般为2 4 6 8 16 32。
   - `m` token数。
   - `k` 激活矩阵的Hidden维。
   - `n` 权重矩阵的Hidden维。
   - `weightNz` 权重矩阵是否是Nz数据格式，默认为1。
   - `dequantGranularity` 反量化的方式，默认为3。
   - `local_expert_nums` 每张卡上的专家数量。
   - `EP` 与peSize保持一致。
   - `maxOutputSize` alltoall后的最大的token数，多余的则截断，默认开两倍的m。
   - `topK` 每个token复制的份数。
   - `transB` 权重矩阵是否转置，默认为0。

   2.2 - **执行生成脚本**:
   
   该步骤会在scripts/run.sh中自动执行，无需单独执行。
   ```bash
   cd examples/dispatch_gmm_combine
   # 基于cpu实现
   python3 utils/gen_data.py
   # 基于torch-npu实现（默认用gen_data.py生成的输入）
   python3 utils/gen_data_by_torch_npu.py
   ```
   注：运行用例需安装torch-npu

3. **运行Dispatch-Gmm-Combine示例程序**
   进入示例目录并执行运行脚本，参数同config中的保持一致：
   ```bash
   cd examples/dispatch_gmm_combine
   bash scripts/run.sh -pes {peSize} -M {m} -K {k} -N {n} -expertPerPe {local_expert_nums} -dataType {dataType} -weightNz {weightNz} -transB {transB}
   ```
   `scripts/run.sh`会执行算子（输出结果保存在`examples/dispatch_gmm_combine/out`目录下）并进行结果校验。
   也可以单独对结果进行校验：
   ```bash
   cd examples/dispatch_gmm_combine
   python3 utils/check_result.py
   ```
   
4. **运行示例**
   ```bash
   # 先将配置写入config.ini
   cd examples/dispatch_gmm_combine
   bash ./scripts/run.sh -pes 2 -M 64 -K 7168 -N 4096 -expertPerPe 2 -dataType 2 -weightNz 1 -transB 0
   ```
