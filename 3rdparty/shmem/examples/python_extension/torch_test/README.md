# shmem算子接入torch样例
该目录提供了shmem部分算子接入torch后的使用效果，仅做展示使用，不建议生产环境使用！
## 编译运行
在shmem根目录执行以下命令：
```sh
# 编译example算子用例及其torch扩展
bash scripts/build.sh -python_example
source install/set_env.sh
cd examples/python_extension/torch_test
python xxx.py # 默认拉起八卡用例
python xxx.py --pes 2 # --pes可以指定卡数，拉起两卡用例
```
注：xxx表示文件名，需要根据实际情况替换。
## 参数说明
### --tool 参数
- `--tool 0`：直接运行，不使用性能分析工具（默认值）
- `--tool 1`：使用 msprof 性能分析工具

**注意**：内存检测工具（如 mssanitizer）和性能采集工具（如 msprof）不能同时使用！

### --pes 参数
- `--pes <N>`：指定使用的卡数，例如 `--pes 2` 表示使用2张卡

## 内存检测工具使用说明
如需使用内存检测工具（mssanitizer），需要：
1. **编译时添加编译选项**：
   ```sh
   bash scripts/build.sh -python_example -mssanitizer
   ```

2. **运行时使用 mssanitizer 拉起**：
   ```sh
   mssanitizer -- python xxx.py --pes 2
   ```
   注意：加完编译选项后，不能直接用 python xx.py 拉起，必须使用 mssanitizer -- python xx.py 方式运行！
