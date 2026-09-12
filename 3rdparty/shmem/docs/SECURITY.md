# 安全声明

## 安全加固
### 加固须知

本文中列出的安全加固措施为基本的加固建议项。用户应根据自身业务，重新审视整个系统的网络安全加固措施。用户应按照所在组织的安全策略进行相关配置，包括并不局限于软件版本、口令复杂度要求、安全配置（协议、加密套件、密钥长度等）， 权限配置、防火墙设置等。必要时可参考业界优秀加固方案和安全专家的建议。
### 通信矩阵

| 组件               | tcp store                                          |
| ------------------ | -------------------------------------------------- |
| 源设备             | tcp client                                         |
| 源IP               | device IP                                          |
| 源端口             | 操作系统自动分配，分配范围由操作系统的自身配置决定 |
| 目的设备           | tcp server                                         |
| 目的IP             | 设备地址IP                                         |
| 目的端口（侦听）   | 用户指定，端口号1025~65535                         |
| 协议               | TCP                                                |
| 端口说明           | server与client TCP协议消息接口                     |
| 侦听端口是否可更改 | 是                                                 |
| 认证方式           | 数字证书认证                                       |
| 加密方式           | TLS 1.3                                            |
| 所属平面           | 管理面                                             |
| 版本               | 所有版本                                           |
| 特殊场景           | 无                                                 |

说明：
支持通过接口 `aclshmemx_set_conf_store_tls` 配置TLS秘钥证书等，进行tls安全连接，建议用户开启TLS加密配置，保证通信安全。系统启动后，建议删除本地密钥证书等信息敏感文件。调用该接口时，传入的文件路径不能包含英文分号、逗号、冒号。
支持通过环境变量 `ACCLINK_CHECK_PERIOD_HOURS`和`ACCLINK_CERT_CHECK_AHEAD_DAYS` 配置证书检查周期与证书过期预警时间

使用接口示例：
```c
// 配置关闭tls:
aclshmemx_set_conf_store_tls(false, nullptr, 0);

// 配置打开tls:

char *tls_info ="                               \
    tlsCaPath: /etc/ssl/certs/;                 \
    tlsCert: /etc/ssl/certs/server.crt;         \
    tlsPk: /etc/ssl/private/server.key;         \
    tlsPkPwd: /etc/ssl/private/key_pwd.txt;     \
    tlsCrlPath: /etc/ssl/crl/;                  \
    tlsCrlFile: server_crl1.pem,server_crl2.pem;\
    tlsCaFile: ca.pem1,ca.pem2;                 \
    packagePath: /etc/lib"
int32_t ret = aclshmemx_set_conf_store_tls(true, tls_info, strlen(tls_info));
```
| 环境变量          | 说明            |
| -------------- | ------------- |
| SHMEM_LOG_LEVEL| shmem日志级别     |
| SHMEM_HOME_PATH| shmem安装路径     |
| VERSION        | 编译whl包默认版本号 |
| ASCEND_RT_VISIBLE_DEVICES | 指定哪些Device对当前进程可见，支持一次指定一个或多个Device ID。通过该环境变量，可实现不修改应用程序即可调整所用Device的功能。 |
| SHMEM_CYCLE_PROF_PE  |  用于设置需要进行Profiling采集的pe，pe_id设置范围[0，PEs-1]，需要取消采集请`unset SHMEM_CYCLE_PROF_PE` |


## 运行用户建议

root是Linux系统中的超级特权用户，具有所有Linux系统资源的访问权限。如果允许直接使用root账号登录Linux系统对系统进行操作，会带来很多潜在的安全风险。通常情况下，建议将在“/etc/ssh/sshd_config”文件中将“PermitRootLogin”参数设置为“no”，设置后root用户无法通过SSH登录到系统，增加了系统的安全性。如果需要使用root权限进行管理操作，可以通过其他普通用户登录系统后，再使用su或sudo命令切换到root用户进行操作。这样可以避免直接使用root用户登录系统，从而减少系统被攻击的风险。出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户使用shmem。

## 文件权限控制

- 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
- 建议用户对个人隐私数据、商业资产、源文件和算子开发过程中保存的各类文件等敏感内容做好权限控制等安全措施。例如涉及本项目安装目录权限管控、输入公共数据文件权限管控，设定的权限建议参考<a href="#file-permission-table">A-文件（夹）各场景权限管控推荐最大值</a>。
- 用户安装和使用过程需要做好权限控制，建议参考<a href="#file-permission-table">A-文件（夹）各场景权限管控推荐最大值</a>文件权限参考进行设置。

## 构建安全声明

在源码编译安装本项目时，需要您自行编译，编译过程中会生成一些中间文件，建议您在编译完成后，对中间文件做好权限控制，以保证文件安全。

## 运行安全声明

- 建议用户结合运行环境资源状况编写对应算子调用脚本。若算子调用脚本与资源状况不匹配，如生成输入数据或标杆计算结果使用空间超出内存容量限制、脚本在本地保存数据超过磁盘空间大小等情况，可能会引发错误并导致进程意外退出。
- 算子在运行异常时会退出进程并打印报错信息，建议根据报错提示定位具体错误原因，包括设定算子同步执行、查看日志文件等方式。
- 算子通过[PyTorch](https://gitcode.com/ascend/pytorch)方式调用时，可能会因为版本不匹配导致运行错误，具体请参考[PyTorch安全声明](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.md)。

## 内存地址随机化机制安全加固

ASLR（address space layout randomization）开启后能增强漏洞攻击防护能力，建议用户将/proc/sys/kernel/randomize_va_space里面的值设置为2，开启该功能。

## 公网地址声明
本项目代码中包含的公网地址声明如下所示：

| 类型  |        开源代码地址        | 文件名                  | 公网IP地址/公网URL地址/域名/邮箱地址/压缩文件地址                                           | 用途说明                                      |
| :---: | :------------------------: | :---------------------- | :------------------------------------------------------------------------------------------ | :-------------------------------------------- |
| 依赖  |           不涉及           | build.sh                | https://gitcode.com/cann/catlass.git                                                            | 从gitcode下载catlass源码，作用编译依赖          |
| 依赖  | https://github.com/doxygen | build.sh                | https://github.com/doxygen/doxygen/releases/download/Release_1_9_6/doxygen-1.9.6.src.tar.gz | 从github下载doxygen-1.9.6源码，作用编译依赖   |
| 依赖  |           不涉及           | build.sh                | https://gitcode.com/GitHub_Trending/go/googletest                                                    | 从gitcode下载googletest源码，单元测试框架依赖   |
| 文档  |           不涉及           | shmemi_device_barrier.h | https://www.inf.ed.ac.uk/teaching/courses/ppls/BarrierPaper.pdf                             | 并行编程语言和系统                            |
## 漏洞机制说明
[漏洞管理](https://gitcode.com/cann/community/blob/master/security/security.md)

## 附录
<a name="file-permission-table"></a>

### A-文件（夹）各场景权限管控推荐最大值

| 类型           | Linux权限参考最大值 |
| -------------- | ---------------  |
| 用户主目录                        |   750（rwxr-x---）            |
| 程序文件(含脚本文件、库文件等)       |   550（r-xr-x---）             |
| 程序文件目录                      |   550（r-xr-x---）            |
| 配置文件                          |  640（rw-r-----）             |
| 配置文件目录                      |   750（rwxr-x---）            |
| 日志文件(记录完毕或者已经归档)        |  440（r--r-----）             |
| 日志文件(正在记录)                |    640（rw-r-----）           |
| 日志文件目录                      |   750（rwxr-x---）            |
| Debug文件                         |  640（rw-r-----）         |
| Debug文件目录                     |   750（rwxr-x---）  |
| 临时文件目录                      |   750（rwxr-x---）   |
| 维护升级文件目录                  |   770（rwxrwx---）    |
| 业务数据文件                      |   640（rw-r-----）    |
| 业务数据文件目录                  |   750（rwxr-x---）      |
| 密钥组件、私钥、证书、密文文件目录    |  700（rwx—----）      |
| 密钥组件、私钥、证书、加密密文        | 600（rw-------）      |
| 加解密接口、加解密脚本            |   500（r-x------）        |