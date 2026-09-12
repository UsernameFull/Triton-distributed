/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDMA_DL_HCCP_DEF_H
#define UDMA_DL_HCCP_DEF_H

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <iostream>
#include <map>
#include "dl_hccp_def.h"

namespace shm {

constexpr int32_t DEV_EID_INFO_MAX_NAME = 64;
constexpr int32_t DEV_QP_KEY_SIZE = 64;
constexpr int32_t HCCP_MAX_TPID_INFO_NUM = 128;
constexpr int32_t CUSTOM_CHAN_DATA_MAX_SIZE = 2048;
constexpr int32_t MAX_INTERFACE_NUM = 8;
constexpr uint32_t TOKEN_VALUE = 0;
constexpr int32_t MEM_KEY_SIZE = 128;
constexpr int32_t URMA_TOKEN_PLAIN_TEXT = 1;

enum DrvHdcServiceType: int {
    HDC_SERVICE_TYPE_RDMA = 6,      /* RDMA Service */
    HDC_SERVICE_TYPE_RDMA_V2 = 18   /* RDMA Service V2, allowing creating server in multi-process at host */
};

struct RaInitConfig {
    unsigned int phyId;             /* 物理Device Id, 第三方网卡当前传0。取值范围[0, 32)*/
    HccpNetworkMode nicPosition;    /* 支持的场景 */
    DrvHdcServiceType hdcType;      /* HDC服务类型 */
    bool enableHdcAsync;            /* 初始化异步HDC session, 用于异步下发指令到device */
};

inline std::ostream &operator<<(std::ostream &output, const RaInitConfig &config)
{
    output << "phyId: " << config.phyId << " nicPosition: " << config.nicPosition << " hdcType: " << config.hdcType
       << " enableHdcAsync: " << config.enableHdcAsync;
    return output;
}

struct MemKey {
    // RDMA: 4Bytes for uint32_t rkey
    // UB: 52Bytes for urma_seg_t seg
    uint8_t value[MEM_KEY_SIZE];
    uint8_t size;
};

inline std::ostream &operator<<(std::ostream &output, const MemKey &memKey)
{
    output << "mem_key: " << std::hex;
    for (auto i = 0U; i < memKey.size; i++) {
        output << "-" << memKey.value[i];
    }
    output << std::dec;
    return output;
}

enum SubProcType {
    TSD_SUB_PROC_HCCP           = 0,    /* hccp process  */
    TSD_SUB_PROC_COMPUTE        = 1,    /* aicpu_schedule process */
    TSD_SUB_PROC_CUSTOM_COMPUTE = 2,    /* aicpu_cust_schedule process */
    TSD_SUB_PROC_QUEUE_SCHEDULE = 3,    /* queue_schedule process */
    TSD_SUB_PROC_UDF            = 4,    /* udf process */
    TSD_SUB_PROC_NPU            = 5,    /* npu process */
    TSD_SUB_PROC_PROXY          = 6,    /* proxy process */
    TSD_SUB_PROC_BUILTIN_UDF    = 7,    /* build in udf */
    TSD_SUB_PROC_ADPROF         = 8,    /* adprof process */
    TSD_SUB_PROC_MAX            = 0xFF
};

struct ProcEnvParam {
    const char* envName;
    uint64_t nameLen;
    const char* envValue;
    uint64_t valueLen;
};

struct ProcExtParam {
    const char* paramInfo;
    uint64_t paramLen;
};

struct ProcOpenArgs {
    SubProcType procType;       /* TSD_SUB_PROC_HCCP; 指定拉起hccp_service.bin进程*/
    ProcEnvParam* envParaList;  /* NULL; 不涉及*/
    uint64_t envCnt;            /* 0 */
    const char* filePath;       /* NULL; 不涉及*/
    uint64_t pathLen;           /* 0 */
    ProcExtParam* extParamList;
    uint64_t extParamCnt;
    pid_t* subPid;              /* 出参 */
};

struct RaInfo {
    HccpNetworkMode mode;
    unsigned int phyId;     /* 物理设备id */
};

inline std::ostream &operator<<(std::ostream &output, const RaInfo &info)
{
    output << "mode: " << info.mode << " phyId: " << info.phyId;
    return output;
}

union HccpEid {
    uint8_t raw[16];        /* Network Order */
    struct {
        uint64_t reserved;  /* If IPv4 mapped to IPv6, == 0 */
        uint32_t prefix;    /* If IPv4 mapped to IPv6, == 0x00000000*/
        uint32_t addr;      /* If IPv4 mapped to IPv6, == IPv4 address */
    } in4;
    struct {
        uint64_t subnetPrefix;
        uint64_t interfaceId;
    } in6;
};

struct DevEidInfo {
    char name[DEV_EID_INFO_MAX_NAME];
    uint32_t type;
    uint32_t eidIndex;
    HccpEid eid;
    uint32_t dieId;
    uint32_t chipId;
    uint32_t funcId;
    uint32_t resv;
};

inline std::ostream &operator<<(std::ostream &output, const DevEidInfo &info)
{
    output << "name: " << info.name << " type: " << info.type << " eidIndex: " << info.eidIndex
           << " eid: " << info.eid.in4.addr << " dieId: " << info.dieId << " chipId: " << info.chipId
           << " funcId: " << info.funcId;
    return output;
}

struct CtxInitCfg {
    HccpNetworkMode mode;
    union {
        struct {
            bool disabledLiteThread;  /* true will not start lite thread */
        } rdma;
    };
};

struct CtxInitAttr {
    unsigned int phyId;             /* physical device id */
    union {
        struct {
            HccpNotifyType notifyType;
            int family;             /* AF_INET(ipv4) of AF_INET6(ipv6) */
            HccpIpAddr localIp;
        } rdma;
        struct {
            uint32_t eidIndex;
            HccpEid eid;
        } ub;
    };
    uint32_t resv[16];
};

inline std::ostream &operator<<(std::ostream &output, const CtxInitAttr &attr)
{
    output << "phyId: " << attr.phyId << " ub: " << attr.ub.eidIndex <<  " eid: " << attr.ub.eid.in4.addr;
    return output;
}

struct DevNotifyInfo {
    uint64_t va;
    uint64_t size;
    MemKey key;
    uint32_t resv[4];
};

struct DevBaseAttrT {
    uint32_t sqMaxDepth;
    uint32_t rqMaxDepth;
    uint32_t sqMaxSge;
    uint32_t rqMaxSge;
    union {
        struct {
            DevNotifyInfo globalNotifyInfo;
        } rdma;
        struct {
            uint32_t maxJfsInlineLen;
            uint32_t maxJfsRsge;
            uint32_t dieId;
            uint32_t chipId;
            uint32_t funcId;
        } ub;
    } devInfo;
    uint32_t resv[16];
};

union DataPlaneCstmFlag {
    struct {
        uint32_t poolCqCstm : 1; // 0: hccp poll cq; 1: caller poll cq
        uint32_t reserved : 31;
    } bs;
    uint32_t value;
};

struct ChanInfoT {
    struct {
        DataPlaneCstmFlag dataPlaneFlag;
    } in;
    struct {
        int fd;
    } out;
};

enum JfcMode {
    JFC_MODE_NORMAL     = 0,        /* Corresponding jetty mode: JETTY_MODE_URMA_NORMAL and JETTY_MODE_USER_CTL_NORMAL */
    JFC_MODE_STARS_POLL = 1,        /* Corresponding jetty mode: JETTY_MODE_CACHE_LOCK_DWQE and JETTY_MODE_USER_CTL_NORMAL */
    JFC_MODE_CCU_POLL   = 2,        /* Corresponding jetty mode: JETTY_MODE_CCU */
    JFC_MODE_USER_CTL_NORMAL = 3,   /* Corresponding jetty mode: JETTY_MODE_USER_CTL_NORMAL */
    JFC_MODE_MAX
};

union JfcFlag {
    struct {
        uint32_t lockFree  : 1;
        uint32_t jfcInline : 1;
        uint32_t reserved  : 30;
    } bs;
    uint32_t value;
};

enum {
    RA_RDMA_NOR_MODE        = 0,
    RA_RDMA_GDR_TMPL_MODE   = 1,
    RA_RDMA_OP_MODE         = 2,
    RA_RDMA_GDR_ASYN_MODE   = 3,
    RA_RDMA_OP_MODE_EXT     = 4,
    RA_RDMA_ERR_MODE        = 5
};

struct CqCreateAttr {
    void* chanHandle;
    uint32_t depth;
    union {
        struct {
            uint64_t cqContext;
            uint32_t mode;          /*refer to enum HCCP_RDMA_NOR_MODE etc*/
            uint32_t compVector;
        } rdma;
        
        struct {
            uint64_t userCtx;
            JfcMode mode;
            uint32_t ceqn;
            JfcFlag flag;           /* refer to urma_jfc_flag_t */
            struct {
                bool valid;
                uint32_t cqeFlag;   /* Indicates whether the jfc is handling the current die or cross-die CCU CQE */
            } ccuExCfg;
        } ub;
    };
};

struct CqCreateInfo {
    uint64_t va; /* refer to struct urma_jfc, struct ibv_cq */
    uint32_t id;
    uint32_t cqeSize;
    uint64_t bufAddr;
    uint64_t swdbAddr;
};

struct CqInfoT {
    CqCreateAttr in;
    CqCreateInfo out;
};

enum JettyMode {
    JETTY_MODE_URMA_NORMAL      = 0, /* jetty_id belongs to [0, 1023] */
    JETTY_MODE_CACHE_LOCK_DWQE  = 1, /* jetty_id belongs to [1216, 5311] */
    JETTY_MODE_CCU              = 2, /* jetty_id belongs to [1024, 1151] */
    JETTY_MODE_USER_CTL_NORMAL  = 3, /* jetty_id belongs to [5312, 9407] */
    JETTY_MODE_CCU_TA_CACHE     = 4, /* jetty_id belongs to [1024, 1151]*/
    JETTY_MODE_MAX
};

enum TransportModeT {
    CONN_RM = 1, /* only for UB, Reliable Message */
    CONN_RC = 2  /* Reliable Connection */
};

union JettyFlag {
    struct {
        uint32_t shareJfr : 1; /* 0: URMA_NO_SHARE_JFR; 1: URMA_SHARE_JFR*/
        uint32_t reserved : 31;
    } bs;
    uint32_t value;
};

union JfsFlag {
    struct {
        uint32_t lockFree       : 1;      /* default as 0, lock protected */
        uint32_t errorSuspend   : 1;      /* 0: error continue; 1: error suspend */
        uint32_t outorderComp   : 1;      /* 0: not support; 1:support out-of-order completion */
        uint32_t orderType      : 8;      /* 
                                           * (0x0): default, auto config by driver
                                           * (0x1): OT, target ordering
                                           * (0x2): OI, initiator ordering
                                           * (0x3): OL, low layer ordering
                                           * (0x4): UNO, unreliable non ordering
                                           */
        uint32_t multiPath      : 1;      /* 1: multi-path, 0: single path, for ubagg only*/
        uint32_t reserved       : 20;
    } bs;
    uint32_t value;
};

struct JettyQueCfgEx {
    uint32_t buffSize;
    uint64_t buffVa;
};

union CstmJfsFlag {
    struct {
        uint32_t sqCstm     : 1;
        uint32_t dbCstm     : 1;
        uint32_t dbCtlCstm  : 1;
        uint32_t reserved   : 29;
    } bs;
    uint32_t value;
};

struct QpCreateAttr {
    void* scqHandle;
    void* rcqHandle;
    void* srqHandle;
    uint32_t sqDepth;
    uint32_t rqDepth;
    TransportModeT transportMode;
    union {
        struct {
            uint32_t mode;          /* refer to RA_RDMA_NOR_MODE etc. */
            uint32_t udpSport;      /* UDP source port */
            uint8_t trafficClass;   /* traffic class */
            uint8_t sl;             /* service level */
            uint8_t timeout;        /* local ack timeout */
            uint8_t rnrRetry;       /* RNR retry count */
            uint8_t retryCnt;       /* retry count */
        } rdm;

        struct {
            JettyMode mode;
            uint32_t jettyId;       /* [optional] user specified jetty id, 0 means not specified */
            JettyFlag flag;
            JfsFlag jfsFlag;        /* refer to union urma_jfs_flag; jfs_cfg->flag */
            void* tokenIdHandle;    /* NULL means unspecified */
            uint32_t tokenValue;    /* refer to urma_token_t; jfr_cfg->token_value */
            uint8_t priority;       /* the priority of JFS. services with low delay need set high priority. Range: [0-0xf] */
            uint8_t rnrRetry;       /* the RNR retry count when receive RNR response; Range: [0, 7]*/
            uint8_t errTimeout;     /* the timeout to report error. Range: [0, 31]*/
            union {
                struct {
                    JettyQueCfgEx sq;       /* specify sq buffer config, required when cstm_flag.bs.sq_cstm specified */
                    bool piType;            /* false: op mode; true: async mode*/
                    CstmJfsFlag cstmFlag;
                    uint32_t sqebbNum;      /* required when cstm_flag.bs.sq_cstm specified */
                } extMode;
                struct {
                    bool lockFlag;          /* false: unlock; true: locked by buffer. forced: true */
                    uint32_t sqeBufIdx;     /* base sqe index*/
                } taCacheMode;
            };
        } ub;
    };
    uint32_t resv[16];
};

struct QpKeyT {
    // ROCE: qpn(4), psn(4), gid(16), tc(4), sl(4), retry_cnt(4), timeout(4) etc
    // UB: jetty_id(24), trans_mode(4)
    uint8_t value[DEV_QP_KEY_SIZE];
    uint8_t size;
};

struct QpCreateInfo {
    QpKeyT key;                     /* for modifying qp or import and bind jetty*/
    union {
        struct {
            uint32_t qpn;
        } rdma;
        struct {
            uint32_t uasid;
            uint32_t id;            /* jetty id*/
            uint64_t sqBuffVa;      /* valida in jetty mode: JETTY_MODE_CACHE_LOCK_DWQE and JETTY_MODE_USER_CLT_NORMAL */
            uint64_t wqebbSize;
            uint64_t dbAddr;
            uint32_t dbTokenId;
            uint64_t ciAddr;
        } ub;
    };
    uint64_t va;                    /* refer to struct urma_jetty*, struct ibv_qp* */
    uint32_t resv[16U];
};

struct HccpTokenId {
    uint32_t tokenId;
};

enum TokenPolicy : uint32_t {
    TOKEN_POLICY_NONE           = 0,
    TOKEN_POLICY_PLAIN_TEXT     = 1,
    TOKEN_POLICY_SIGNED         = 2,
    TOKEN_POLICY_ALL_ENCRYPTED  = 3,
    TOKEN_POLICY_RESERVED
};

union ImportJettyFlag {
    struct {
        uint32_t tokenPolicy : 3;   /* refer to enum TokenPolicy */
        uint32_t orderType   : 8;   /* 
                                     * (0x0): default, auto config by driver
                                     * (0x1): OT, target ordering
                                     * (0x2): OI, initiator ordering
                                     * (0x3): OL, low layer ordering
                                     * (0x4): UNO, unreliable non ordering
                                     */
        uint32_t shareTp     : 1;   /* 1: shared tp; 0: non-shared tp. when rc mode is not ta dst ordering, this flag can only be set to 0. */
        uint32_t reserved    : 20;
    } bs;
    uint32_t value;
};

enum JettyGrpPolicy : uint32_t {
    JETTY_GRP_POLICY_RR        = 0,
    JETTY_GRP_POLICY_HASH_HINT = 1,
    JETTY_GRP_POLICY_MAX
};

enum TargetType {
    TARGET_TYPE_JFR         = 0,
    TARGET_TYPE_JETTY       = 1,
    TARGET_TYPE_JETTY_GROUP = 2,
    TARGET_TYPE_MAX
};

enum JettyImportMode {
    JETTY_IMPORT_MODE_NORMAL = 0,   /* 普通模式 */
    JETTY_IMPORT_MODE_EXP    = 1,   /* 扩展模式MAMI建链 */
    JETTY_IMPORT_MODE_MAX
};

struct JettyImportExpCfg {
    uint64_t tpHandle;
    uint64_t peerTpHandle;
    uint64_t tag;
    uint32_t txPsn;
    uint32_t rxPsn;
    uint32_t rsv[16];
};

struct QpImportAttr {
    QpKeyT key;                                 /* for RDMA, save key on rem_qp_handle for bind to modify qp*/
    union {
        struct {
            JettyImportMode mode;
            uint32_t tokenValue;                /* refer to urma_token_t */
            JettyGrpPolicy policy;
            TargetType type;
            ImportJettyFlag flag;
            JettyImportExpCfg expImportCfg;     /* only valid on mode JETTY_IMPORT_MODE_EXP */
            uint32_t tpType;                    /* refer to urma_tp_type_t */
        } ub;
    };
    uint32_t resv[7];
};

struct QpImportInfo {
    union {
        struct {
            uint64_t tjettyHandle; /* refer to urma_target_t *tjetty */
            uint32_t tpn;          /* refer to urma_tp_t tp*/
        } ub;
    };
    uint32_t resv[8];
};

struct QpImportInfoT {
    QpImportAttr in;
    QpImportInfo out;
};

enum MemSegTokenPolicy {
    MEM_SEG_TOKEN_NONE          = 0,
    MEM_SEG_TOKEN_PLAIN_TEXT    = 1,
    MEM_SEG_TOKEN_SIGNED        = 2,
    MEM_SEG_TOKEN_ALL_ENCRYPTED = 3,
    MEM_SEG_TOKEN_RESERVED      = 4
};

struct MemInfo {
    uint64_t addr;
    uint64_t size;
};

union RegSegFlag {
    struct {
        uint32_t tokenPolicy    : 3;   /* refer to enum MemSegTokenPolicy */
        uint32_t cacheable      : 1;   /* 0: URMA_NON_CACHABLE; 1: URMA_CACHEABLE */
        uint32_t dsva           : 1;    
        uint32_t access         : 6;   /* refer to enum MemSegAccessFlags */
        uint32_t nonPin         : 1;   /* 0: segment pages pined; 1: segment pages non-pined */
        uint32_t userIova       : 1;   /* 0: segment without user iova ddr; 1: segment with user iova ddr */
        uint32_t tokenIdValid   : 1;   /* 0: token id in cfg is invalid; 1: token id in cfg is valid */
        uint32_t reserved       : 18;
    } bs;
    uint32_t value;
};

struct MemRegAttr {
    MemInfo mem;
    union {
        struct {
            int access;          /* refer to enum MemSegAccessFlags */
        } rdma;
        struct {
            RegSegFlag flags;
            uint32_t tokenValue; /* refer to urma_token_t */
            void* tokenIdHandle; /* NULL means unspecified, valid if flags.token_id_valid has been set*/
        } ub;
    };
    uint32_t resv[8];
};

struct MemRegInfo {
    MemKey key;
    union {
        struct {
            uint32_t lkey;
        } rdma;
        struct {
            uint32_t tokenId;
            uint64_t targetSegHandle;   /* refer to urma_target_seg_t */
        } ub;
    };
    uint32_t resv[8U];
};

struct MrRegInfoT {
    MemRegAttr in;
    MemRegInfo out;
};

union ImportSegFlag {
    struct {
        uint32_t cacheable : 1;     /* 0: URMA_NONE_CACHEABLE; 1: URMA_CACHEABLE*/
        uint32_t access    : 6;     /* refer to enum MemSegAccessFlag */
        uint32_t mapping   : 1;     /* 0: URMA_SEG_NOMAP; 1: URMA_SEG_MAPPED*/
        uint32_t reserved  : 24;
    } bs;
    uint32_t value;
};

struct MemImportAttr {
    MemKey key;
    union {
        struct {
            ImportSegFlag flags;
            uint64_t mappingAddr;   /* addr is needed if flag mapping set value */
            uint32_t tokenValue;    /* refer to urma_token_t */
        } ub;
    };
    uint32_t resv[4];
};

struct MemImportInfo {
    union {
        struct {
            uint32_t key;
        } rdma;
        struct {
            uint64_t targetSegHandle;  /* refer to urma_target_seg_t */
        } ub;
    };
    uint32_t resv[4];
};

struct MrImportInfoT {
    MemImportAttr in;
    MemImportInfo out;
};

enum JettyAttrMask : uint32_t {
    JETTY_ATTR_RX_THRESHOLD = 0x01,
    JETTY_ATTR_STATE = (0x01 << 1)
};

enum JettyState {
    JETTY_STATE_RESET = 0,
    JETTY_STATE_READY,
    JETTY_STATE_SUSPENDED,
    JETTY_STATE_ERROR
};

struct JettyAttr {
    JettyAttrMask mask;
    uint32_t rxThreshold;
    JettyState state;
    uint32_t resv[2];
};

struct WrSgeList {
    uint64_t addr;
    uint32_t len;
    void* lmemHandle;
};

enum RaWrOpcode {
    RA_WR_RDMA_WRITE,
    RA_WR_RDMA_WRITE_WITH_IMM,
    RA_WR_SEND,
    RA_WR_SEND_WITH_IMM,
    RA_WR_RDMA_READ,
    RA_WR_RDMA_ATOMIC_WRITE = 0xf0,
    RA_WR_RDMA_WRITE_WITH_NOTIFY = 0xf2,
    RA_WR_RDMA_REDUCE_WRITE = 0xf5,
    RA_WR_RDMA_REDUCE_WRITE_WITH_NOTIFY = 0xf6
};

struct WrAuxInfo {
    uint8_t dataType;
    uint8_t reduceType;
    uint32_t notifyOffset;
};

struct WrNotifyInfo {
    uint64_t notifyData;    /* notify data */
    uint64_t notifyAddr;    /* remote notify addr */
    void* notifyHandle;     /* remote notify handle*/
};

struct WrReduceInfo {
    bool reduceEn;
    uint8_t reduceOpcode;
    uint8_t reduceDataType;
};

enum RaUbOpcode {
    RA_UB_OPC_WRITE         = 0x00,
    RA_UB_OPC_WRITE_NOTIFY  = 0x02,
    RA_UB_OPC_READ          = 0x10,
    RA_UB_OPC_NOP           = 0x51,
    RA_UB_OPC_LAST          = 0x00,
};

union JfsWrFlag {
    struct {
        uint32_t placeOrder : 2;        /*
                                         * 0: There is no order with other WR.
                                         * 1: relax order
                                         * 2: strong order
                                         * 3: reverse. see urma_order_type_t
                                         */
        uint32_t compOrder : 1;         /* 
                                         * 0: There is no completion order with other WR.
                                         * 1: Completion order with previous WR.
                                         */
        uint32_t fence : 1;             /* 
                                         * 0: There is no fence
                                         * 1: Fence with previous read and atomic WR.
                                         */
        uint32_t solicitedEnable : 1;   /* 0: not solicited; 1: Solicited */
        uint32_t completeEnable : 1;    /* 0: Do not Generate CR for this WR; 1: Generate CR for this WR after the WR is completed */
        uint32_t inlineFlag : 1;        /* 0: Not inline data; 1: Inline data*/
        uint32_t reserved : 25;
    } bs;
    uint32_t value;
};

struct SendWrData {
    WrSgeList* sges;
    uint32_t numSge;                    /* size of segs, not exceeds to MAX_SGE_NUM */
    uint8_t* inlineData;
    uint32_t inlineSize;                /* size of inline_data, see struct dev_base_attr_t */
    uint64_t remoteAddr;
    void* rmemHandle;
    union {
        struct {
            uint64_t wrId;
            RaWrOpcode opcode;
            unsigned int flags;         /* reference to ra_send_flags */
            WrAuxInfo aux;              /* aux info */
        } rdma;
        struct {
            uint64_t userCtx;
            RaUbOpcode opcode;          
            JfsWrFlag flags;
            void* rem_qp_handle;        /* resv for RM use */
            WrNotifyInfo notifyInfo;    /* required for opcode RA_UB_OPC_WRITE_NOTIFY */
            WrReduceInfo reduceInfo;    /* inline reduce is enabled when reduce_en is set to true */
        } ub;
    };
    uint32_t immData;
    uint32_t resv[10];
};

struct UbPostInfo {
    uint16_t funcId : 7;
    uint16_t dieId  : 1;
    uint16_t rsv    : 8;
    uint16_t jettyId;
    uint16_t piVal;         /* doorbell value */
    uint8_t dwqe[128];      /* direct wqe */
    uint16_t dwqeSize;      /* size of dwqe calc by piVal, 64 or 128 */
};

struct WqeInfoT {
    unsigned int sqIndex;
    unsigned int wqeIndex;
};

struct DbInfo {
    unsigned int dbIndex;
    unsigned long dbInfo;
};

struct SendWrResp {
    union {
        WqeInfoT wqeTmp;            /* wqe template info used for V80 offload */
        DbInfo db;                  /* doorbell info used for V71 and V80 opbase */
        UbPostInfo doorbellInfo;    /* doorbell info used for UB */
        uint8_t resv[384];          /* resv fro write value doorbell info */
    };
};

struct CustomChanInfoIn {
    char data[CUSTOM_CHAN_DATA_MAX_SIZE];
    unsigned int offsetStart;
    unsigned int op;
};

struct CustomChanInfoOut {
    char data[CUSTOM_CHAN_DATA_MAX_SIZE];
    unsigned int offsetNext;
    int opRet;
};

union GetTpCfgFlag {
    struct {
        uint32_t ctp            : 1;
        uint32_t rtp            : 1;
        uint32_t utp            : 1;
        uint32_t uboe           : 1;
        uint32_t preDefined     : 1;
        uint32_t dynamicDefined : 1;
        uint32_t reserved       : 26;
    } bs;
    uint32_t value;
};

struct GetTpCfg {
    GetTpCfgFlag flag;
    TransportModeT transMode;
    HccpEid localEid;
    HccpEid peerEid;
};

struct TpInfo {
    uint64_t tpHandle;
    uint32_t resv;
};

struct RaGetIfAttr {
    unsigned int phyId;         /* physical device id */
    HccpNetworkMode nicPosition;    /* reference to NetworkMode */
    bool isAll;                 /* 
                                 * valid when nic_position is NETWORK_FFLINE. 
                                 * false: get specific rnic ip
                                 * true: get all rnic ip 
                                 */
};

struct IfAddrInfo {
    HccpIpAddr ip;
    struct in_addr mask;
};

struct InterfaceInfo {
    int family;
    int scopeId;
    IfAddrInfo ifAddr;   /* Address and netmask of interface */
    char ifName[MAX_INTERFACE_NUM];      /* Name of interface */
};

enum MemSegAccessFlags {
    MEM_SEG_ACCESS_LOCAL_ONLY = 1,
    MEM_SEG_ACCESS_READ = (1 << 1),
    MEM_SEG_ACCESS_WRITE = (1 << 2),
    MEM_SEG_ACCESS_ATOMIC = (1 << 3),
    MEM_SEG_ACCESS_DEFAULT = MEM_SEG_ACCESS_READ | MEM_SEG_ACCESS_WRITE | MEM_SEG_ACCESS_ATOMIC,
};

struct RegMemResultInfo {
    uint32_t reserved{0};
    uint64_t address{0};
    uint64_t size{0};
    void* lmemHandle{nullptr};
    struct MemKey key{{0}, 0};
    uint32_t tokenId{0};
    uint32_t tokenValue{0};
    uint64_t targetSegHandle{0};
    void* tokenIdHandle{nullptr};
    uint32_t cacheable{0};
    int32_t access{0};

    RegMemResultInfo() :
        reserved(0),
        address(0),
        size(0),
        lmemHandle(nullptr),
        key{{0}, 0},
        tokenId(0),
        tokenValue(0),
        targetSegHandle(0),
        tokenIdHandle(nullptr),
        cacheable(0),
        access(0)
    {
    }

    RegMemResultInfo(
        uint64_t addr, 
        uint64_t sz, 
        void* handle,
        struct MemKey& memKey,
        uint32_t id,
        uint32_t value,
        uint64_t targetHandle,
        void* idHandle,
        uint32_t isCacheable,
        int32_t acc
    ) : reserved(0), 
        address(addr),
        size(sz),
        lmemHandle(handle),
        key(memKey),
        tokenId(id),
        tokenValue(value),
        targetSegHandle(targetHandle),
        tokenIdHandle(idHandle),
        cacheable(isCacheable),
        access(acc)
    {
    }

    friend inline std::ostream& operator<<(std::ostream& output, const RegMemResultInfo& info)
    {
        output << "Address: " << info.address << ", Size: " << info.size << ", key: " << info.key
               << ", tokenId: " << info.tokenId << ", tokenValue: " << info.tokenValue
               << ", targetSegHandle: " << info.targetSegHandle << ", cacheable: " << info.cacheable
               << ", access: " << info.access;
        return output;
    }
};

using MemoryRegionMap = std::map<uint64_t, std::map<uint32_t, RegMemResultInfo>, std::greater<uint64_t>>;


} // namespace shm

#endif // !UDMA_DL_HCCP_API_H
