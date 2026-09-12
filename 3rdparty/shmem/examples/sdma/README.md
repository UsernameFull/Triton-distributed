# SDMA使用说明
## 环境要求和准备
SDMA功能在9.0.0及以上版本（尝鲜版）新增支持。需要下载并安装以下cann和ops软件包：
- toolkit包（[CANN master obp镜像网站](https://mirror-centralrepo.devcloud.cn-north-4.huaweicloud.com/artifactory/cann-run-mirror/software/master/)）
- ops-legacy包（根据硬件平台下载对应版本：[A2 x86_64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260520_newest/cann-910b-ops-legacy_9.1.0_linux-x86_64.run)/[A2 aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260520_newest/cann-910b-ops-legacy_9.1.0_linux-aarch64.run)/[A3 x86_64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260520_newest/cann-A3-ops-legacy_9.1.0_linux-x86_64.run)/[A3 aarch64](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260520_newest/cann-A3-ops-legacy_9.1.0_linux-aarch64.run)）

## example使用方式：
1.在`shmem/`目录编译软件包并安装：
```bash
bash scripts/build.sh -package
./install/*/SHMEM_1.0.0_linux-*.run --install
```

2.在`shmem/`目录下编译examples：
```bash
bash scripts/build.sh -examples
```

3.在`shmem/examples/sdma`目录执行demo:
```bash
bash run.sh -pes ${PES} -type ${TYPES}
````
  - **参数说明**：
      - PES：指定用于运行的设备（NPU）数量，限定单台机器内。
      - TYPES：指定传输数据类型，当前支持：int，uint8，int64，fp32。

## SDMA接口使用说明

### aclshmemx_sdma_put_nbi
以指针类型参数接口为例：
```c++
ACLSHMEM_DEVICE void aclshmemx_sdma_put_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                            uint32_t elem_size, int pe, uint32_t sync_id)
```
接口功能：把PE pe上的src地址中的数据传输到dst地址，传输elem_size个元素。
| 参数名       | 含义                                                                 |
|--------------|----------------------------------------------------------------------|
| dst          | 目标卡上目的地址在本卡上的对称地址                                   |
| src          | 本卡上的源地址                                                       |
| buf          | 缓冲区地址                                                           |
| ub_size      | 缓冲区大小                                                           |
| elem_size    | 元素个数                                                             |
| pe           | 目标PE                                                               |
| sync_id      | 同步ID                                                               |

### aclshmemx_sdma_get_nbi
以指针类型参数接口为例：
```c++
ACLSHMEM_DEVICE void aclshmemx_sdma_get_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf, uint32_t ub_size,
                                            uint32_t elem_size, int pe, uint32_t sync_id)
```
接口功能：把PE pe上的dst地址中的数据传输到src地址，传输elem_size个元素。
| 参数名       | 含义                                                                 |
|--------------|----------------------------------------------------------------------|
| dst          | 目标卡上目的地址在本卡上的对称地址                                   |
| src          | 本卡上的源地址                                                       |
| buf          | 缓冲区地址                                                           |
| ub_size      | 缓冲区大小                                                           |
| elem_size    | 元素个数                                                             |
| pe           | 目标PE                                                               |
| sync_id      | 同步ID                                                               |

## 注意事项
`aclshmemx_sdma_put_nbi`和`aclshmemx_sdma_get_nbi`都是非阻塞接口，调用后立即返回，不等待数据传输完成。用户使用时，可通过以下两种方式确保数据传输完成：
1. 所有调用`aclshmemx_sdma_put/get_nbi`的核，在sdma任务结束后，算子内调用`aclshmemx_sdma_quiet`接口，等待所有SDMA操作完成。  
适用场景：算子内后续操作依赖sdma任务完成，例如后续算子需要使用sdma传输好的数据。
2. 所有调用`aclshmemx_sdma_put/get_nbi`的核，在sdma任务结束后，算子内调用`aclshmemx_sdma_notify_record`接口，然后在host侧调用`aclrtWaitAndResetNotify`接口，等待指定的同步ID完成（详细用法可查看[NotifyWait机制使用说明](../notifywait/README.md)）。  
适用场景：其它stream上的kernel需要等待sdma任务完成后才能继续执行。
