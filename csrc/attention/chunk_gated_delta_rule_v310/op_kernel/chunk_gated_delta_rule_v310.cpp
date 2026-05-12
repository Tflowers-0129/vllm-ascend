#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "chunk_gated_delta_rule_v310_tiling_data.h"
#define UB_SIZE 262144
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ceil_div(a, b) (a+b-1)/b

using namespace AscendC;
using namespace ChunkGatedDeltaRuleV310;

class ChunkGatedDeltaRuleV310Kernel {
public:
    __aicore__ inline ChunkGatedDeltaRuleV310Kernel(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR g, GM_ADDR beta, GM_ADDR core_attn, GM_ADDR last_recurrent_state,
                                GM_ADDR workspace, const ChunkGatedDeltaRuleTilingData &tilingIn, TPipe *pipeIn);
    __aicore__ inline void InitUB();
    __aicore__ inline void InitWorkSpace(GM_ADDR workspace);
    template <bool Cast_Flag = true>
    __aicore__ inline void L2normDim128Float(LocalTensor<half> src, uint32_t head_num);
    __aicore__ inline void L2normDim128FloatTrans(LocalTensor<half> src, uint32_t head_num);
    __aicore__ inline void Transpose_64_128();
    __aicore__ inline void TransposeBetaHalf();
    __aicore__ inline void TransposeGFloat();
    template <bool Zero_Diag = false>
    __aicore__ inline void Inv_bisec16_opt(LocalTensor<float> dst, LocalTensor<float> src);
    template <bool Zero_Diag = false>
    __aicore__ inline void Inv_bisec16(LocalTensor<float> dst, LocalTensor<float> src);
    __aicore__ inline void Inv_bisec32_2(LocalTensor<float> dst, LocalTensor<float> src);
    __aicore__ inline void ND2NZ_32x32(LocalTensor<float> dst, LocalTensor<float> src);
    __aicore__ inline void NZ2ND_32x32(LocalTensor<float> dst, LocalTensor<float> src);
    __aicore__ inline void NZ2ZZ_32x32(LocalTensor<half> dst, LocalTensor<half> src);
    __aicore__ inline void HighPrecCast(LocalTensor<half> dst, LocalTensor<float> src, int size);
    __aicore__ inline void HighPrecLoad_32x32_A(LocalTensor<half> dst, LocalTensor<half> src);
    __aicore__ inline void HighPrecLoad_32x32_B(LocalTensor<half> dst, LocalTensor<half> src);

    __aicore__ inline void HighPrecLoad_16x16x4_A(LocalTensor<half> dst, LocalTensor<half> src, int repeatTimes, int srcStride, int offset);
    __aicore__ inline void HighPrecLoad_16x16x4_B(LocalTensor<half> dst, LocalTensor<half> src, int repeatTimes, int srcStride, int offset);

    __aicore__ inline void GCumsumFloat();
    __aicore__ inline void Process();

    __aicore__ inline void LoadQKVHalf(LocalTensor<half> dst, GlobalTensor<half> src, uint32_t head_dim, uint32_t block_count);
    __aicore__ inline void StoreAttnHalf(GlobalTensor<half> dst, LocalTensor<half> src, uint32_t head_dim, uint32_t block_count);
    __aicore__ inline void LoadQKVHalfWithTail(LocalTensor<half> dst, GlobalTensor<half> src, uint32_t head_dim, uint32_t chunk_index);
    __aicore__ inline void StoreAttnHalfWithTail(GlobalTensor<half> dst, LocalTensor<half> src, uint32_t head_dim, uint32_t chunk_index);

    template <int Event_ID = 0>
    __aicore__ inline void UB2L1_ND2NZ(LocalTensor<half> dst, LocalTensor<half> src, LocalTensor<half> tmp, uint16_t row, uint16_t column);
    template <bool Trans>
    __aicore__ inline void L12L0_NZ(LocalTensor<half> dst, LocalTensor<half> src, uint16_t row, uint16_t column);
    template <bool Trans>
    __aicore__ inline void L12L0_ZN(LocalTensor<half> dst, LocalTensor<half> src, uint16_t row, uint16_t column);
    __aicore__ inline void L0C2UB_NZ2ND(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<float> tmp, uint16_t row, uint16_t column);
    template <int Event_ID = 1, bool TransposeA = false, bool TransposeB = false, bool LoadA = true, bool LoadB = true>
    __aicore__ inline void MatMul(LocalTensor<float> dst, LocalTensor<half> srcA, LocalTensor<half> srcB, LocalTensor<float> tmp, uint16_t m, uint16_t n, uint16_t k, int offsetC = 0, int offsetA = 0, int offsetB = 0);

    GlobalTensor<half> qGlobal;
    GlobalTensor<half> kGlobal;
    GlobalTensor<half> vGlobal;
    GlobalTensor<float> gGlobal;
    GlobalTensor<half> bGlobal;
    GlobalTensor<float> isGlobal;

    GlobalTensor<half> attnGlobal;
    GlobalTensor<float> lsGlobal;
    const ChunkGatedDeltaRuleTilingData *tiling;

    GlobalTensor<float> gtransGlobal;
    GlobalTensor<half> btransGlobal;
    GlobalTensor<float> gcumsumGlobal;

    TPipe *pipe;
    uint32_t block_id;
    uint32_t block_num;

    LocalTensor<half> l1_temp0;
    LocalTensor<half> l0a_temp0;
    LocalTensor<half> l0b_temp0;
    LocalTensor<float> l0c_temp0;

    LocalTensor<float> ub_temp0;
    LocalTensor<float> ub_temp1;
    LocalTensor<float> ub_temp2;
    LocalTensor<float> ub_temp3;
    LocalTensor<float> ub_temp4;
    LocalTensor<float> ub_temp5;
    LocalTensor<float> ub_temp6;
    LocalTensor<float> ub_temp7;
    LocalTensor<float> ub_temp8;
    LocalTensor<float> ub_temp9;
    LocalTensor<float> ub_temp10;
    LocalTensor<float> ub_temp11;
    LocalTensor<float> ub_temp12;

    LocalTensor<float> I;
    LocalTensor<float> I2;

    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t numHead;
    uint32_t headDimQK;
    uint32_t headDimV;
    uint32_t seqLenPaded;
    uint32_t chunkSize;

    float headDimQKfp32;
    uint32_t singleCoreNumHead;
    uint32_t tailSeqLen;

    uint64_t inputPtr[16];
    uint64_t outputPtr[16];

    uint64_t inputPtr2[16];
    uint64_t outputPtr2[16];
};

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR g, GM_ADDR beta, GM_ADDR core_attn, GM_ADDR last_recurrent_state,
                                                           GM_ADDR workspace, const ChunkGatedDeltaRuleTilingData &tilingIn, TPipe *pipeIn) {
    pipe = pipeIn;
    block_id = GetBlockIdx();
    block_num = GetBlockNum();

    tiling = &tilingIn;

    batchSize = tiling->batchSize;
    seqLen = tiling->seqLen;
    numHead = tiling->numHead;
    headDimQK = tiling->headDimQK;
    headDimV = tiling->headDimV;
    chunkSize = 64;
    singleCoreNumHead = numHead / block_num;
    tailSeqLen = seqLen % chunkSize;
    seqLenPaded = ceil_div(seqLen, chunkSize) * chunkSize;

    uint64_t sizeQK = batchSize * seqLen * numHead * headDimQK;
    uint64_t sizeV = batchSize * seqLen * numHead * headDimV;
    qGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(query), sizeQK);
    kGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(key), sizeQK);
    vGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(value), sizeV);
    uint64_t sizeGBeta = batchSize * seqLen * numHead;
    gGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(g), sizeGBeta);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(beta), sizeGBeta);
    uint64_t sizeState = batchSize * numHead * headDimQK * headDimV;
    attnGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(core_attn), sizeV);
    lsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(last_recurrent_state), sizeState);

    InitUB();
    InitWorkSpace(workspace);
    LocalTensor<int32_t> ub_temp0_int32_t = ub_temp0.ReinterpretCast<int32_t>();
    PipeBarrier<PIPE_ALL>();
    ub_temp0_int32_t.SetValue(0, headDimQK);
    PipeBarrier<PIPE_ALL>();
    Cast(ub_temp1, ub_temp0_int32_t, RoundMode::CAST_NONE, 8);
    PipeBarrier<PIPE_ALL>();
    headDimQKfp32 = ub_temp1.GetValue(0);

    inputPtr[0] = reinterpret_cast<uint64_t>(ub_temp6.GetPhyAddr());
    for (int i = 1; i < 16; i++) {
        inputPtr[i] = inputPtr[i - 1] + 32;
    }
    outputPtr[0] = reinterpret_cast<uint64_t>(ub_temp12.GetPhyAddr());
    outputPtr[1] = outputPtr[0] + 32;
    for (int i = 2; i < 16; i++) {
        outputPtr[i] = outputPtr[i - 2] + 128 * 64 / 8 * 4;
    }

    inputPtr2[0] = reinterpret_cast<uint64_t>(ub_temp1.GetPhyAddr());
    inputPtr2[8] = reinterpret_cast<uint64_t>(ub_temp1[32].GetPhyAddr());
    for (int i = 1; i < 8; i++) {
        inputPtr2[i] = inputPtr2[0];
        inputPtr2[i + 8] = inputPtr2[8];
    }
    outputPtr2[0] = reinterpret_cast<uint64_t>(ub_temp12.GetPhyAddr());
    outputPtr2[1] = outputPtr2[0] + 32 * 8 * 4;
    for (int i = 2; i < 16; i++) {
        outputPtr2[i] = outputPtr2[i - 2] + 32;
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::InitUB() {
    TBuf<TPosition::VECCALC> UbBuf;
    TBuf<TPosition::A2> L0ABuf;
    TBuf<TPosition::B2> L0BBuf;
    TBuf<TPosition::A1> L1Buf;
    TBuf<TPosition::CO1> L0CBuf;

    pipe->InitBuffer(UbBuf, UB_SIZE);

    ub_temp0 = UbBuf.Get<float>();
    ub_temp1 = ub_temp0[64 * 64]; //1
    ub_temp2 = ub_temp1[64 * 64]; //2
    ub_temp3 = ub_temp2[64 * 64]; //3
    ub_temp4 = ub_temp3[64 * 64]; //4
    ub_temp5 = ub_temp4[64 * 64]; //5
    ub_temp6 = ub_temp5[64 * 64]; //6
    ub_temp7 = ub_temp6[64 * 64]; //7

    ub_temp8 = ub_temp7[64 * 64]; //8
    ub_temp9 = ub_temp8[64 * 64]; //9
    ub_temp10 = ub_temp9[64 * 64]; //10
    ub_temp11 = ub_temp10[64 * 128]; //12
    ub_temp12 = ub_temp11[64 * 128]; //14

    I = ub_temp1[64 * 16 - 16 * 16];
    I2 = ub_temp1[64 * 16 - 16 * 16 * 2];

    pipe->InitBuffer(L1Buf, 1024 * 1024);
    l1_temp0 = L1Buf.Get<half>();

    pipe->InitBuffer(L0ABuf, 1024 * 64);
    l0a_temp0 = L0ABuf.Get<half>();

    pipe->InitBuffer(L0BBuf, 1024 * 64);
    l0b_temp0 = L0BBuf.Get<half>();

    pipe->InitBuffer(L0CBuf, 1024 * 256);
    l0c_temp0 = L0CBuf.Get<float>();
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::InitWorkSpace(GM_ADDR workspace) {
    btransGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(workspace),
                                 batchSize * seqLenPaded * numHead);
    gtransGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + batchSize * seqLenPaded * numHead,
                                 batchSize * seqLenPaded * numHead);
    gcumsumGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + batchSize * seqLenPaded * numHead * 2,
                                  batchSize * seqLenPaded * numHead);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::LoadQKVHalf(LocalTensor<half> dst,
                                                                  GlobalTensor<half> src,
                                                                  uint32_t head_dim,
                                                                  uint32_t block_count) {
    DataCopyParams repeatParams_half_qkv;
    repeatParams_half_qkv.blockLen = head_dim * 2 / 32;
    repeatParams_half_qkv.srcGap = head_dim * (numHead - 1) * 2 / 32;
    repeatParams_half_qkv.dstGap = 0;
    repeatParams_half_qkv.blockCount = block_count;
    DataCopy<half>(dst, src, repeatParams_half_qkv);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::LoadQKVHalfWithTail(LocalTensor<half> dst,
                                                                          GlobalTensor<half> src,
                                                                          uint32_t head_dim,
                                                                          uint32_t chunk_index) {
    if ((chunk_index == ((seqLenPaded / chunkSize) - 1)) && (tailSeqLen != 0)) {
        Duplicate<half>(dst, (half)0, chunkSize * head_dim);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        LoadQKVHalf(dst, src, head_dim, tailSeqLen);
    } else {
        LoadQKVHalf(dst, src, head_dim, chunkSize);
    }
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::StoreAttnHalf(GlobalTensor<half> dst,
                                                                    LocalTensor<half> src,
                                                                    uint32_t head_dim,
                                                                    uint32_t block_count) {
    DataCopyParams repeatParams_half_attn;
    repeatParams_half_attn.blockLen = head_dim * 2 / 32;
    repeatParams_half_attn.srcGap = 0;
    repeatParams_half_attn.dstGap = head_dim * (numHead - 1) * 2 / 32;
    repeatParams_half_attn.blockCount = block_count;
    DataCopy<half>(dst, src, repeatParams_half_attn);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::StoreAttnHalfWithTail(GlobalTensor<half> dst,
                                                                              LocalTensor<half> src,
                                                                              uint32_t head_dim,
                                                                              uint32_t chunk_index) {
    if ((chunk_index == ((seqLenPaded / chunkSize) - 1)) && (tailSeqLen != 0)) {
        StoreAttnHalf(dst, src, head_dim, tailSeqLen);
    } else {
        StoreAttnHalf(dst, src, head_dim, chunkSize);
    }
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::Transpose_64_128() {
    for (int i = 0; i < 16; i++) {
        DataCopy<float>(ub_temp6[64 * 8 * i], ub_temp12[8 * i], {64, 1, 15, 0});
    }

    TransDataTo5HDParams nchwconvParams;
    nchwconvParams.repeatTimes = 128 * 64 / 16 / 8;
    nchwconvParams.dstRepStride = 2;
    nchwconvParams.srcRepStride = 16;

    TransDataTo5HD<float>(outputPtr, inputPtr, nchwconvParams);

    for (int i = 0; i < 16; i++) {
        DataCopy<float>(ub_temp6[64 * 8 * i], ub_temp12[64 * i], {8, 8, 120, 0});
    }
}

template <bool Cast_Flag>
__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::L2normDim128Float(LocalTensor<half> src, uint32_t head_num) {
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    Cast(ub_temp6, src, RoundMode::CAST_NONE, head_num * headDimQK);
    DataCopy<float>(ub_temp12, ub_temp6, head_num * headDimQK);
    Mul<float>(ub_temp12, ub_temp6, ub_temp12, head_num * headDimQK);

    RepeatReduceSum<float>(ub_temp11[4096], ub_temp12, head_num * 2, 64, 0, 1, 1, 8);

    if (head_num > 32) {
        PairReduceSum<float>(ub_temp12, ub_temp11[4096], 1, 32 * 2, 1, 1, 8);
        PairReduceSum<float>(ub_temp12[32], ub_temp11[4096 + 32 * 2], 1, (head_num - 32) * 2, 1, 1, 8);
    } else {
        PairReduceSum<float>(ub_temp12, ub_temp11[4096], 1, head_num * 2, 1, 1, 8);
    }

    Adds<float>(ub_temp12, ub_temp12, (float)0.000001, head_num);

    Sqrt<float>(ub_temp12, ub_temp12, head_num);
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);

    for (int k = 0; k < head_num; k++) {
        float temp = ub_temp12.GetValue(k);
        Muls<float>(ub_temp6[k * headDimQK], ub_temp6[k * headDimQK], ((float)1.0) / temp, headDimQK);
    }
    if constexpr (Cast_Flag) {
        Cast(src, ub_temp6, RoundMode::CAST_ODD, head_num * headDimQK);
    }
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::L2normDim128FloatTrans(LocalTensor<half> src, uint32_t head_num) {
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    Cast(ub_temp12, src, RoundMode::CAST_NONE, head_num * headDimQK);
    Mul<float>(ub_temp6, ub_temp12, ub_temp12, head_num * headDimQK);

    RepeatReduceSum<float>(ub_temp11[4096], ub_temp6, head_num * 2, 64, 0, 1, 1, 8);

    if (head_num > 32) {
        PairReduceSum<float>(ub_temp6, ub_temp11[4096], 1, 32 * 2, 1, 1, 8);
        PairReduceSum<float>(ub_temp6[32], ub_temp11[4096 + 32 * 2], 1, (head_num - 32) * 2, 1, 1, 8);
    } else {
        PairReduceSum<float>(ub_temp6, ub_temp11[4096], 1, head_num * 2, 1, 1, 8);
    }

    Adds<float>(ub_temp6, ub_temp6, (float)0.000001, head_num);

    Sqrt<float>(ub_temp1[64 * 16], ub_temp6, head_num);

    Transpose_64_128();
    Duplicate(ub_temp12, (float)1, 64);
    Div(ub_temp1[64 * 16], ub_temp12, ub_temp1[64 * 16], head_num);
    Mul<float>(ub_temp6, ub_temp6, ub_temp1[64 * 16], 64, headDimQK, {1, 1, 1, 8, 8, 0});
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::GCumsumFloat() {
    LocalTensor<uint8_t> stackBuffer = ub_temp11.ReinterpretCast<uint8_t>();

    uint32_t totalChunk = batchSize * numHead * (seqLenPaded / chunkSize);
    uint32_t chunkBatch = 256;
    uint32_t chunkRepeat = ceil_div(totalChunk, chunkBatch);
    uint32_t chunkTail = totalChunk % chunkBatch;
    uint32_t dataLength;
    uint32_t ub_temp_size = 64 * 64 * 6;
    uint32_t numChunk;
    uint32_t g_index = 0;
    for (int r = 0; r < chunkRepeat; r++) {
        if ((r == (chunkRepeat - 1)) && (chunkTail != 0)) {
            numChunk = chunkTail;
        } else {
            numChunk = chunkBatch;
        }
        dataLength = numChunk * chunkSize;
        uint32_t dataLength = numChunk * chunkSize;

        TransposeParamsExt transposeParamsIn;
        transposeParamsIn.nSize = 1;
        transposeParamsIn.cSize = numChunk;
        transposeParamsIn.hSize = 1;
        transposeParamsIn.wSize = chunkSize;
        transposeParamsIn.transposeType = TransposeType::TRANSPOSE_NCHW2NHWC;

        TransposeParamsExt transposeParamsOut;
        transposeParamsOut.nSize = 1;
        transposeParamsOut.cSize = chunkSize;
        transposeParamsOut.hSize = 1;
        transposeParamsOut.wSize = numChunk;
        transposeParamsOut.transposeType = TransposeType::TRANSPOSE_NCHW2NHWC;

        DataCopy<float>(ub_temp0, gtransGlobal[g_index], dataLength);

        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        Transpose(ub_temp6, ub_temp0, stackBuffer, transposeParamsIn);

        Duplicate<float>(ub_temp4, (float)0, chunkSize * chunkSize * 2);

        for (int i = 0; i < 6; i++) {
            if (dataLength >= 2048) {
                uint32_t repeatTimes = ceil_div(dataLength, 2048);
                for (int j = 0; j < repeatTimes; j++) {
                    Add(ub_temp0[j * 2048], ub_temp0[ub_temp_size + j * 2048], ub_temp0[ub_temp_size + j * 2048 - (1 << i) * numChunk], 2048);
                }
                DataCopy<float>(ub_temp0[ub_temp_size], ub_temp0, dataLength);
            } else {
                Add(ub_temp0[ub_temp_size], ub_temp0[ub_temp_size], ub_temp0[ub_temp_size - (1 << i) * numChunk], dataLength);
            }
        }

        Transpose(ub_temp0, ub_temp6, stackBuffer, transposeParamsOut);

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        DataCopy<float>(gcumsumGlobal[g_index], ub_temp0, dataLength);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
        g_index = g_index + dataLength;
    }
}

template <int Event_ID>
__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::UB2L1_ND2NZ(LocalTensor<half> dst, LocalTensor<half> src, LocalTensor<half> tmp, uint16_t row, uint16_t column) {
    for (int i = 0; i < column / 16; i++) {
        DataCopy(tmp[16 * row * i], src[16 * i], {row, 1, static_cast<uint16_t>(column / 16 - 1), 0});
    }
    SetFlag<HardEvent::V_MTE3>(Event_ID);
    WaitFlag<HardEvent::V_MTE3>(Event_ID);
    DataCopy(dst, tmp, row * column);
}

template <bool Trans>
__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::L12L0_NZ(LocalTensor<half> dst, LocalTensor<half> src, uint16_t row, uint16_t column) {
    LoadData2DParams loadData2DParams;
    loadData2DParams.repeatTimes = row / 16;
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = column / 16 - 1;
    loadData2DParams.ifTranspose = Trans;
    for (int i = 0; i < column / 16; i++) {
        LoadData(dst[16 * 16 * i], src[16 * row * i], loadData2DParams);
    }
}

template <bool Trans>
__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::L12L0_ZN(LocalTensor<half> dst, LocalTensor<half> src, uint16_t row, uint16_t column) {
    LoadData2DParams loadData2DParams;
    loadData2DParams.repeatTimes = row * column / (16 * 16);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = Trans;
    LoadData(dst, src, loadData2DParams);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::L0C2UB_NZ2ND(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<float> tmp, uint16_t row, uint16_t column) {
    DataCopyEnhancedParams enhancedParams;
    enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(tmp, src, {1, static_cast<uint16_t>(row * column / (16 * 16 * 2)), 0, 0}, enhancedParams);
    for (int i = 0; i < column / (16 * 2); i++) {
        DataCopy(dst[16 * i], tmp[16 * row * i], {row, 2, 0, static_cast<uint16_t>(column / 8 - 2)});
    }
    DataCopy(tmp, src[row * column / 2], {1, static_cast<uint16_t>(row * column / (16 * 16 * 2)), 0, 0}, enhancedParams);
    for (int i = 0; i < column / (16 * 2); i++) {
        DataCopy(dst[16 * i + column / 2], tmp[16 * row * i], {row, 2, 0, static_cast<uint16_t>(column / 8 - 2)});
    }
}

template <int Event_ID, bool TransposeA, bool TransposeB, bool LoadA, bool LoadB>
__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::MatMul(LocalTensor<float> dst, LocalTensor<half> srcA, LocalTensor<half> srcB, LocalTensor<float> tmp, uint16_t m, uint16_t n, uint16_t k, int offsetC, int offsetA, int offsetB) {
    if constexpr (LoadA) {
        if constexpr (TransposeA) {
            L12L0_ZN<true>(l0a_temp0[offsetA], srcA, m, k);
        } else {
            L12L0_NZ<false>(l0a_temp0[offsetA], srcA, m, k);
        }
    }
    WaitFlag<HardEvent::MTE3_MTE1>(Event_ID);
    if constexpr (LoadB) {
        if constexpr (TransposeB) {
            L12L0_ZN<false>(l0b_temp0[offsetB], srcB, k, n);
        } else {
            L12L0_NZ<true>(l0b_temp0[offsetB], srcB, k, n);
        }
    }
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.cmatrixInitVal = true;
    Mmad(l0c_temp0[offsetC], l0a_temp0[offsetA], l0b_temp0[offsetB], mmadParams);
    SetFlag<HardEvent::M_V>(0);
    WaitFlag<HardEvent::M_V>(0);
    L0C2UB_NZ2ND(dst, l0c_temp0[offsetC], tmp, m, n);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::ND2NZ_32x32(LocalTensor<float> dst, LocalTensor<float> src) {
    DataCopy<float>(dst, src, {32, 2, 6, 0});
    DataCopy<float>(dst[16 * 32], src[16], {32, 2, 6, 0});
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::NZ2ND_32x32(LocalTensor<float> dst, LocalTensor<float> src) {
    DataCopy<float>(dst, src, {32, 2, 0, 6});
    DataCopy<float>(dst[16], src[16 * 32], {32, 2, 0, 6});
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::HighPrecLoad_32x32_A(LocalTensor<half> dst, LocalTensor<half> src) {
    LoadData2DParams loadData2DParamsA;
    loadData2DParamsA.repeatTimes = 2;
    loadData2DParamsA.srcStride = 1;
    loadData2DParamsA.dstGap = 0;
    loadData2DParamsA.ifTranspose = false;
    LoadData(dst, src, loadData2DParamsA);
    LoadData(dst[96 * 16], src[16 * 32], loadData2DParamsA);
    LoadData(dst[16 * 32], src[32 * 32], loadData2DParamsA);
    LoadData(dst[96 * 16 + 16 * 32], src[32 * 32 + 16 * 32], loadData2DParamsA);
    LoadData(dst[16 * 32 * 2], src, loadData2DParamsA);
    LoadData(dst[96 * 16 + 16 * 32 * 2], src[16 * 32], loadData2DParamsA);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::HighPrecLoad_16x16x4_A(LocalTensor<half> dst, LocalTensor<half> src, int repeatTimes, int srcStride, int offset) {
    LoadData2DParams loadData2DParamsA;
    loadData2DParamsA.repeatTimes = repeatTimes;
    loadData2DParamsA.srcStride = srcStride;
    loadData2DParamsA.dstGap = 2;
    loadData2DParamsA.ifTranspose = false;
    LoadData(dst, src, loadData2DParamsA);
    LoadData(dst[16 * 16], src[offset], loadData2DParamsA);
    LoadData(dst[16 * 16 * 2], src, loadData2DParamsA);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::HighPrecLoad_16x16x4_B(LocalTensor<half> dst, LocalTensor<half> src, int repeatTimes, int srcStride, int offset) {
    LoadData2DParams loadData2DParamsB;
    loadData2DParamsB.repeatTimes = repeatTimes;
    loadData2DParamsB.srcStride = srcStride;
    loadData2DParamsB.dstGap = 2;
    loadData2DParamsB.ifTranspose = true;
    LoadData(dst, src, loadData2DParamsB);
    LoadData(dst[16 * 16], src, loadData2DParamsB);
    LoadData(dst[16 * 16 * 2], src[offset], loadData2DParamsB);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::HighPrecLoad_32x32_B(LocalTensor<half> dst, LocalTensor<half> src) {
    LoadData2DParams loadData2DParamsB;
    loadData2DParamsB.repeatTimes = 4;
    loadData2DParamsB.srcStride = 1;
    loadData2DParamsB.dstGap = 0;
    loadData2DParamsB.ifTranspose = true;
    LoadData(dst, src, loadData2DParamsB);
    LoadData(dst[32 * 32], src, loadData2DParamsB);
    LoadData(dst[32 * 32 * 2], src[32 * 32], loadData2DParamsB);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::NZ2ZZ_32x32(LocalTensor<half> dst, LocalTensor<half> src) {
    DataCopy(dst, src, {2, 16, 16, 0});
    DataCopy(dst[16 * 16 * 2], src[16 * 16], {2, 16, 16, 0});
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::HighPrecCast(LocalTensor<half> dst, LocalTensor<float> src, int size) {
    Cast(dst, src, RoundMode::CAST_ODD, size);
    Cast(ub_temp10, dst, RoundMode::CAST_NONE, size);
    Sub(ub_temp10, src, ub_temp10, size);
    Cast(dst[size], ub_temp10, RoundMode::CAST_NONE, size);
}

template <bool Zero_Diag>
__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::Inv_bisec16(LocalTensor<float> dst, LocalTensor<float> src) {
    MmadParams mmadParams;
    mmadParams.m = 16;
    mmadParams.n = 16;
    mmadParams.k = 48;
    mmadParams.cmatrixInitVal = true;

    DataCopyEnhancedParams enhancedParams;
    enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;

    LocalTensor<half> ub_temp_half_9 = ub_temp9.ReinterpretCast<half>();
    LocalTensor<float> baseBlock = ub_temp8[32 * 32 * 2];
    LocalTensor<float> bisecBlock16 = ub_temp8[32 * 32 * 3];

    DataCopy<float>(baseBlock, src, {16, 2, 6, 0});
    DataCopy<float>(baseBlock[16 * 16], src[64 * 16 + 16], {16, 2, 6, 0});
    DataCopy<float>(baseBlock[16 * 16 * 2], src[(64 * 16 + 16) * 2], {16, 2, 6, 0});
    DataCopy<float>(baseBlock[16 * 16 * 3], src[(64 * 16 + 16) * 3], {16, 2, 6, 0});
    if constexpr (Zero_Diag) {
        for (int j = 0; j < 4; j++) {
            Add(baseBlock[16 * 16 * j], baseBlock[16 * 16 * j], I, 16 * 16);
        }
    }
    Muls(baseBlock, baseBlock, (float)-1, 16 * 16 * 4);
    HighPrecCast(ub_temp_half_9, baseBlock, 16 * 16 * 4);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopy(l1_temp0, ub_temp_half_9, 16 * 16 * 4 * 2);
    SetFlag<HardEvent::MTE3_MTE1>(1);
    WaitFlag<HardEvent::MTE3_MTE1>(1);
    HighPrecLoad_16x16x4_A(l0a_temp0[16 * 16 * 3 * 4], l1_temp0, 4, 1, 16 * 16 * 4);
    SetFlag<HardEvent::MTE1_M>(1);
    WaitFlag<HardEvent::MTE1_M>(1);
    for (int j = 0; j < 4; j++) {
        Add(baseBlock[16 * 16 * j], baseBlock[16 * 16 * j], I2, 16 * 16);
    }

    for (int i = 0; i < 3; i++) {
        HighPrecCast(ub_temp_half_9, baseBlock, 16 * 16 * 4);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        {
            if (i == 0) {
                Duplicate(ub_temp8[16 * 16 * 2], (float)0, 16 * 16);
            } else if (i == 1) {
                Duplicate(ub_temp8[16 * 16 * 6], (float)0, 16 * 16);
            } else if (i == 2) {
            }
        }
        DataCopy(l1_temp0[16 * 16 * 4 * 2], ub_temp_half_9, 16 * 16 * 4 * 2);
        SetFlag<HardEvent::MTE3_MTE1>(0);
        WaitFlag<HardEvent::MTE3_MTE1>(0);
        HighPrecLoad_16x16x4_B(l0b_temp0, l1_temp0[16 * 16 * 4 * 2], 4, 1, 16 * 16 * 4);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        for (int j = 0; j < 4; j++) {
            Mmad(l0c_temp0[16 * 16 * j], l0a_temp0[16 * 16 * 3 * 4 + 16 * 16 * 3 * j], l0b_temp0[16 * 16 * 3 * j], mmadParams);
        }
        SetFlag<HardEvent::M_V>(0);
        WaitFlag<HardEvent::M_V>(0);
        DataCopy(baseBlock, l0c_temp0, {1, 4, 0, 0}, enhancedParams);

        for (int j = 0; j < 4; j++) {
            Add(baseBlock[16 * 16 * j], baseBlock[16 * 16 * j], I2, 16 * 16);
        }
        HighPrecCast(ub_temp_half_9, baseBlock, 16 * 16 * 4);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        {
            if (i == 0) {
                DataCopy<float>(bisecBlock16, src[64 * 16], {16, 2, 6, 0});
            } else if (i == 1) {
                DataCopy<float>(bisecBlock16[16 * 16], src[64 * 48 + 32], {16, 2, 6, 0});
            } else if (i == 2) {
                Muls(bisecBlock16, bisecBlock16, (float)-1, 16 * 16 * 2);
            }
        }
        DataCopy(l1_temp0[16 * 16 * 4 * 2 * 2], ub_temp_half_9, 16 * 16 * 4 * 2);
        SetFlag<HardEvent::MTE3_MTE1>(0);
        WaitFlag<HardEvent::MTE3_MTE1>(0);
        HighPrecLoad_16x16x4_A(l0a_temp0, l1_temp0[16 * 16 * 4 * 2], 4, 1, 16 * 16 * 4);
        HighPrecLoad_16x16x4_B(l0b_temp0, l1_temp0[16 * 16 * 4 * 2 * 2], 4, 1, 16 * 16 * 4);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        for (int j = 0; j < 4; j++) {
            Mmad(l0c_temp0[16 * 16 * j], l0a_temp0[16 * 16 * 3 * j], l0b_temp0[16 * 16 * 3 * j], mmadParams);
        }
        SetFlag<HardEvent::M_V>(0);
        WaitFlag<HardEvent::M_V>(0);
        DataCopy(baseBlock, l0c_temp0, {1, 4, 0, 0}, enhancedParams);
    }
    HighPrecCast(ub_temp_half_9, baseBlock, 16 * 16 * 6);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopy(l1_temp0, ub_temp_half_9, 16 * 16 * 6 * 2);
    {
        DataCopy(ub_temp8, baseBlock, 16 * 16);
        DataCopy(ub_temp8[16 * 16 * 3], baseBlock[16 * 16], 16 * 16);
        DataCopy(ub_temp8[16 * 16 * 4], baseBlock[16 * 16 * 2], 16 * 16);
        DataCopy(ub_temp8[16 * 16 * 7], baseBlock[16 * 16 * 3], 16 * 16);
    }
    SetFlag<HardEvent::MTE3_MTE1>(0);
    WaitFlag<HardEvent::MTE3_MTE1>(0);
    HighPrecLoad_16x16x4_A(l0a_temp0, l1_temp0[16 * 16 * 4], 2, 1, 16 * 16 * 6);
    HighPrecLoad_16x16x4_B(l0b_temp0, l1_temp0, 2, 2, 16 * 16 * 6);
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    {
        HighPrecLoad_16x16x4_A(l0a_temp0[16 * 16 * 3 * 2], l1_temp0[16 * 16], 2, 2, 16 * 16 * 6);
    }
    for (int j = 0; j < 2; j++) {
        Mmad(l0c_temp0[16 * 16 * j], l0a_temp0[16 * 16 * 3 * j], l0b_temp0[16 * 16 * 3 * j], mmadParams);
    }
    SetFlag<HardEvent::M_V>(0);
    WaitFlag<HardEvent::M_V>(0);
    DataCopy(bisecBlock16, l0c_temp0, {1, 2, 0, 0}, enhancedParams);

    HighPrecCast(ub_temp_half_9, bisecBlock16, 16 * 16 * 2);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    {
        DataCopy<float>(ub_temp8[32 * 32 * 2], src[64 * 32], {32, 2, 6, 0});
        DataCopy<float>(ub_temp8[32 * 32 * 2 + 16 * 32], src[64 * 32 + 16], {32, 2, 6, 0});
    }
    DataCopy(l1_temp0[16 * 16 * 4], ub_temp_half_9, 16 * 16 * 2);
    DataCopy(l1_temp0[16 * 16 * 6 + 16 * 16 * 4], ub_temp_half_9[16 * 16 * 2], 16 * 16 * 2);
    SetFlag<HardEvent::MTE3_MTE1>(0);
    WaitFlag<HardEvent::MTE3_MTE1>(0);
    HighPrecLoad_16x16x4_B(l0b_temp0, l1_temp0[16 * 16 * 4], 2, 1, 16 * 16 * 6);
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    for (int j = 0; j < 2; j++) {
        Mmad(l0c_temp0[16 * 16 * j], l0a_temp0[16 * 16 * 3 * 2 + 16 * 16 * 3 * j], l0b_temp0[16 * 16 * 3 * j], mmadParams);
    }
    SetFlag<HardEvent::M_V>(0);
    WaitFlag<HardEvent::M_V>(0);
    DataCopy(ub_temp8[16 * 16], l0c_temp0, {1, 1, 0, 0}, enhancedParams);
    DataCopy(ub_temp8[16 * 16 * 5], l0c_temp0[16 * 16], {1, 1, 0, 0}, enhancedParams);
    Inv_bisec32_2(src, ub_temp8);
}


template <bool Zero_Diag>
__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::Inv_bisec16_opt(LocalTensor<float> dst, LocalTensor<float> src) {
    LocalTensor<half> ub_temp_half_4 = ub_temp4.ReinterpretCast<half>();
    MmadParams mmadParams;
    mmadParams.m = 16;
    mmadParams.n = 16;
    mmadParams.k = 48;
    mmadParams.cmatrixInitVal = true;

    DataCopyEnhancedParams enhancedParams;
    enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;

    LocalTensor<half> ub_temp_half_9 = ub_temp9.ReinterpretCast<half>();
    LocalTensor<float> baseBlock = ub_temp8[32 * 32 * 2];
    LocalTensor<float> bisecBlock16 = ub_temp8[32 * 32 * 3];

    DataCopy<float>(baseBlock, src, {16, 2, 6, 0});
    DataCopy<float>(baseBlock[16 * 16], src[64 * 16 + 16], {16, 2, 6, 0});
    DataCopy<float>(baseBlock[16 * 16 * 2], src[(64 * 16 + 16) * 2], {16, 2, 6, 0});
    DataCopy<float>(baseBlock[16 * 16 * 3], src[(64 * 16 + 16) * 3], {16, 2, 6, 0});
    if constexpr (Zero_Diag) {
        for (int j = 0; j < 4; j++) {
            Add(baseBlock[16 * 16 * j], baseBlock[16 * 16 * j], I, 16 * 16);
        }
    }
    Muls(baseBlock, baseBlock, (float)-1, 16 * 16 * 4);
    HighPrecCast(ub_temp_half_9, baseBlock, 16 * 16 * 4);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopy(l1_temp0, ub_temp_half_9, 16 * 16 * 4 * 2);
    SetFlag<HardEvent::MTE3_MTE1>(1);
    WaitFlag<HardEvent::MTE3_MTE1>(1);
    HighPrecLoad_16x16x4_A(l0a_temp0[16 * 16 * 3 * 4], l1_temp0, 4, 1, 16 * 16 * 4);
    SetFlag<HardEvent::MTE1_M>(1);
    WaitFlag<HardEvent::MTE1_M>(1);
    for (int j = 0; j < 4; j++) {
        Add(baseBlock[16 * 16 * j], baseBlock[16 * 16 * j], I2, 16 * 16);
    }
    for (int i = 0; i < 3; i++) {
        Cast(ub_temp_half_9, baseBlock, RoundMode::CAST_ODD, 16 * 16 * 4);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        {
            if (i == 0) {
                Duplicate(ub_temp8[16 * 16 * 2], (float)0, 16 * 16);
                Duplicate(ub_temp8[16 * 16 * 6], (float)0, 16 * 16);
            } else if (i == 1) {
                Add(ub_temp2, ub_temp2, I, 64, 8, {8, 8, 2, 64 + 1, 64 + 1, 0});
            } else if (i == 2) {
                Mul<float>(ub_temp6, ub_temp6, ub_temp2[(chunkSize - 1) * chunkSize], 64, 64, {1, 1, 1, 8, 8, 0});
            }
        }
        DataCopy(l1_temp0[16 * 16 * 4 * 2], ub_temp_half_9, 16 * 16 * 4);
        SetFlag<HardEvent::MTE3_MTE1>(0);
        WaitFlag<HardEvent::MTE3_MTE1>(0);
        {
            LoadData2DParams loadData2DParamsB;
            loadData2DParamsB.repeatTimes = 4;
            loadData2DParamsB.srcStride = 1;
            loadData2DParamsB.dstGap = 2;
            loadData2DParamsB.ifTranspose = true;
            LoadData(l0b_temp0, l1_temp0[16 * 16 * 4 * 2], loadData2DParamsB);
            LoadData(l0b_temp0[16 * 16], l1_temp0[16 * 16 * 4 * 2], loadData2DParamsB);
        }
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        {
            MmadParams mmadParams;
            mmadParams.m = 16;
            mmadParams.n = 16;
            mmadParams.k = 32;
            mmadParams.cmatrixInitVal = true;
            for (int j = 0; j < 4; j++) {
                Mmad(l0c_temp0[16 * 16 * j], l0a_temp0[16 * 16 * 3 * 4 + 16 * 16 * 3 * j], l0b_temp0[16 * 16 * 3 * j], mmadParams);
            }
        }
        SetFlag<HardEvent::M_V>(0);
        WaitFlag<HardEvent::M_V>(0);
        DataCopy(baseBlock, l0c_temp0, {1, 4, 0, 0}, enhancedParams);

        for (int j = 0; j < 4; j++) {
            Add(baseBlock[16 * 16 * j], baseBlock[16 * 16 * j], I2, 16 * 16);
        }
        Cast(ub_temp_half_9, baseBlock, RoundMode::CAST_ODD, 16 * 16 * 4);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        {
            if (i == 0) {
                DataCopy<float>(bisecBlock16, src[64 * 16], {16, 2, 6, 0});
                DataCopy<float>(bisecBlock16[16 * 16], src[64 * 48 + 32], {16, 2, 6, 0});
            } else if (i == 1) {
                Muls(bisecBlock16, bisecBlock16, (float)-1, 16 * 16 * 2);
            } else if (i == 2) {
                Mul<float>(ub_temp6[64 * 64], ub_temp6[64 * 64], ub_temp2[(chunkSize - 1) * chunkSize], 64, 64, {1, 1, 1, 8, 8, 0});
            }
        }
        DataCopy(l1_temp0[16 * 16 * 4 * 2 * 2], ub_temp_half_9, 16 * 16 * 4);
        SetFlag<HardEvent::MTE3_MTE1>(0);
        WaitFlag<HardEvent::MTE3_MTE1>(0);
        {
            LoadData2DParams loadData2DParamsA;
            loadData2DParamsA.repeatTimes = 4;
            loadData2DParamsA.srcStride = 1;
            loadData2DParamsA.dstGap = 2;
            loadData2DParamsA.ifTranspose = false;
            LoadData(l0a_temp0, l1_temp0[16 * 16 * 4 * 2], loadData2DParamsA);
        }
        {
            LoadData2DParams loadData2DParamsB;
            loadData2DParamsB.repeatTimes = 4;
            loadData2DParamsB.srcStride = 1;
            loadData2DParamsB.dstGap = 2;
            loadData2DParamsB.ifTranspose = true;
            LoadData(l0b_temp0, l1_temp0[16 * 16 * 4 * 2 * 2], loadData2DParamsB);
        }
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        {
            MmadParams mmadParams;
            mmadParams.m = 16;
            mmadParams.n = 16;
            mmadParams.k = 16;
            mmadParams.cmatrixInitVal = true;
            for (int j = 0; j < 4; j++) {
                Mmad(l0c_temp0[16 * 16 * j], l0a_temp0[16 * 16 * 3 * j], l0b_temp0[16 * 16 * 3 * j], mmadParams);
            }
        }
        SetFlag<HardEvent::M_V>(0);
        WaitFlag<HardEvent::M_V>(0);
        DataCopy(baseBlock, l0c_temp0, {1, 4, 0, 0}, enhancedParams);
    }

    for (int i = 0; i < 1; i++) {
        HighPrecCast(ub_temp_half_9, baseBlock, 16 * 16 * 4);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        {
            Cast(ub_temp_half_4, ub_temp6, RoundMode::CAST_ODD, chunkSize * 64);
        }
        DataCopy(l1_temp0[16 * 16 * 4 * 2], ub_temp_half_9, 16 * 16 * 4 * 2);
        SetFlag<HardEvent::MTE3_MTE1>(0);
        WaitFlag<HardEvent::MTE3_MTE1>(0);
        HighPrecLoad_16x16x4_B(l0b_temp0, l1_temp0[16 * 16 * 4 * 2], 4, 1, 16 * 16 * 4);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        for (int j = 0; j < 4; j++) {
            Mmad(l0c_temp0[16 * 16 * j], l0a_temp0[16 * 16 * 3 * 4 + 16 * 16 * 3 * j], l0b_temp0[16 * 16 * 3 * j], mmadParams);
        }
        SetFlag<HardEvent::M_V>(0);
        WaitFlag<HardEvent::M_V>(0);
        DataCopy(baseBlock, l0c_temp0, {1, 4, 0, 0}, enhancedParams);

        for (int j = 0; j < 4; j++) {
            Add(baseBlock[16 * 16 * j], baseBlock[16 * 16 * j], I2, 16 * 16);
        }
        Cast(ub_temp_half_9, baseBlock, RoundMode::CAST_ODD, 16 * 16 * 4);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        {
            Cast(ub_temp_half_4[chunkSize * 64], ub_temp6[chunkSize * 64], RoundMode::CAST_ODD, chunkSize * 64); // ub_temp_half_4 = k_i * decay_mask_i[-1]
        }
        DataCopy(l1_temp0[16 * 16 * 4 * 2 * 2], ub_temp_half_9, 16 * 16 * 4);
        SetFlag<HardEvent::MTE3_MTE1>(0);
        WaitFlag<HardEvent::MTE3_MTE1>(0);
        {
           LoadData2DParams loadData2DParamsA;
            loadData2DParamsA.repeatTimes = 4;
            loadData2DParamsA.srcStride = 1;
            loadData2DParamsA.dstGap = 2;
            loadData2DParamsA.ifTranspose = false;
            LoadData(l0a_temp0, l1_temp0[16 * 16 * 4 * 2], loadData2DParamsA);
            LoadData(l0a_temp0[16*16], l1_temp0[16 * 16 * 4 * 3], loadData2DParamsA);
        }
        {
            LoadData2DParams loadData2DParamsB;
            loadData2DParamsB.repeatTimes = 4;
            loadData2DParamsB.srcStride = 1;
            loadData2DParamsB.dstGap = 2;
            loadData2DParamsB.ifTranspose = true;
            LoadData(l0b_temp0, l1_temp0[16 * 16 * 4 * 2 * 2], loadData2DParamsB);
            LoadData(l0b_temp0[16 * 16], l1_temp0[16 * 16 * 4 * 2], loadData2DParamsB);
        }
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        {
            MmadParams mmadParams;
            mmadParams.m = 16;
            mmadParams.n = 16;
            mmadParams.k = 32;
            mmadParams.cmatrixInitVal = true;
            for (int j = 0; j < 4; j++) {
                Mmad(l0c_temp0[16 * 16 * j], l0a_temp0[16 * 16 * 3 * j], l0b_temp0[16 * 16 * 3 * j], mmadParams);
            }
        }
        SetFlag<HardEvent::M_V>(0);
        WaitFlag<HardEvent::M_V>(0);
        DataCopy(baseBlock, l0c_temp0, {1, 4, 0, 0}, enhancedParams);
    }
    HighPrecCast(ub_temp_half_9, baseBlock, 16 * 16 * 6);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopy(l1_temp0, ub_temp_half_9, 16 * 16 * 6 * 2);
    {
        DataCopy(ub_temp8, baseBlock, 16 * 16);
        DataCopy(ub_temp8[16 * 16 * 3], baseBlock[16 * 16], 16 * 16);
        DataCopy(ub_temp8[16 * 16 * 4], baseBlock[16 * 16 * 2], 16 * 16);
        DataCopy(ub_temp8[16 * 16 * 7], baseBlock[16 * 16 * 3], 16 * 16);
    }
    SetFlag<HardEvent::MTE3_MTE1>(0);
    WaitFlag<HardEvent::MTE3_MTE1>(0);
    HighPrecLoad_16x16x4_A(l0a_temp0, l1_temp0[16 * 16 * 4], 2, 1, 16 * 16 * 6);
    HighPrecLoad_16x16x4_B(l0b_temp0, l1_temp0, 2, 2, 16 * 16 * 6);
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    {
        HighPrecLoad_16x16x4_A(l0a_temp0[16 * 16 * 3 * 2], l1_temp0[16 * 16], 2, 2, 16 * 16 * 6);
    }
    for (int j = 0; j < 2; j++) {
        Mmad(l0c_temp0[16 * 16 * j], l0a_temp0[16 * 16 * 3 * j], l0b_temp0[16 * 16 * 3 * j], mmadParams);
    }
    SetFlag<HardEvent::M_V>(0);
    WaitFlag<HardEvent::M_V>(0);
    DataCopy(bisecBlock16, l0c_temp0, {1, 2, 0, 0}, enhancedParams);

    HighPrecCast(ub_temp_half_9, bisecBlock16, 16 * 16 * 2);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    {
        DataCopy<float>(ub_temp8[32 * 32 * 2], src[64 * 32], {32, 2, 6, 0});
        DataCopy<float>(ub_temp8[32 * 32 * 2 + 16 * 32], src[64 * 32 + 16], {32, 2, 6, 0});
    }
    DataCopy(l1_temp0[16 * 16 * 4], ub_temp_half_9, 16 * 16 * 2);
    DataCopy(l1_temp0[16 * 16 * 6 + 16 * 16 * 4], ub_temp_half_9[16 * 16 * 2], 16 * 16 * 2);
    SetFlag<HardEvent::MTE3_MTE1>(0);
    WaitFlag<HardEvent::MTE3_MTE1>(0);
    HighPrecLoad_16x16x4_B(l0b_temp0, l1_temp0[16 * 16 * 4], 2, 1, 16 * 16 * 6);
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    for (int j = 0; j < 2; j++) {
        Mmad(l0c_temp0[16 * 16 * j], l0a_temp0[16 * 16 * 3 * 2 + 16 * 16 * 3 * j], l0b_temp0[16 * 16 * 3 * j], mmadParams);
    }
    SetFlag<HardEvent::M_V>(0);
    WaitFlag<HardEvent::M_V>(0);
    DataCopy(ub_temp8[16 * 16], l0c_temp0, {1, 1, 0, 0}, enhancedParams);
    DataCopy(ub_temp8[16 * 16 * 5], l0c_temp0[16 * 16], {1, 1, 0, 0}, enhancedParams);

    Inv_bisec32_2(src, ub_temp8);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::Inv_bisec32_2(LocalTensor<float> dst, LocalTensor<float> src) {
    LocalTensor<half> ub_temp_half_9 = ub_temp9.ReinterpretCast<half>();
    DataCopyEnhancedParams enhancedParams;
    enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    MmadParams mmadParams;
    mmadParams.m = 32;
    mmadParams.n = 32;
    mmadParams.k = 96;
    mmadParams.cmatrixInitVal = true;

    Muls(src[32 * 32 * 2], src[32 * 32 * 2], (float)-1, 32 * 32);
    HighPrecCast(ub_temp_half_9, src, 32 * 32 * 3);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    NZ2ZZ_32x32(l1_temp0[32 * 32 * 0], ub_temp_half_9);
    NZ2ZZ_32x32(l1_temp0[32 * 32 * 2], ub_temp_half_9[32 * 32]);
    NZ2ZZ_32x32(l1_temp0[32 * 32 * 4], ub_temp_half_9[32 * 32 * 2]);

    NZ2ZZ_32x32(l1_temp0[32 * 32 * 1], ub_temp_half_9[32 * 32 * 3]);
    NZ2ZZ_32x32(l1_temp0[32 * 32 * 3], ub_temp_half_9[32 * 32 * 4]);
    NZ2ZZ_32x32(l1_temp0[32 * 32 * 5], ub_temp_half_9[32 * 32 * 5]);
    {
        NZ2ND_32x32(dst, src[32 * 32 * 0]);
    }

    SetFlag<HardEvent::MTE3_MTE1>(0);
    WaitFlag<HardEvent::MTE3_MTE1>(0);
    HighPrecLoad_32x32_A(l0a_temp0, l1_temp0[32 * 32 * 4]);
    HighPrecLoad_32x32_B(l0b_temp0, l1_temp0[32 * 32 * 0]);
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    {
        HighPrecLoad_32x32_A(l0a_temp0[32 * 32 * 3], l1_temp0[32 * 32 * 2]);
    }
    Mmad(l0c_temp0, l0a_temp0, l0b_temp0, mmadParams);
    SetFlag<HardEvent::M_V>(0);
    WaitFlag<HardEvent::M_V>(0);
    DataCopy(src[32 * 32 * 2], l0c_temp0, {1, 4, 0, 0}, enhancedParams);

    HighPrecCast(ub_temp_half_9[32 * 32 * 2], src[32 * 32 * 2], 32 * 32);
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    NZ2ZZ_32x32(l1_temp0[32 * 32 * 4], ub_temp_half_9[32 * 32 * 2]);
    NZ2ZZ_32x32(l1_temp0[32 * 32 * 5], ub_temp_half_9[32 * 32 * 3]);
    {
        NZ2ND_32x32(dst[64 * 32 + 32], src[32 * 32 * 1]);
    }
    SetFlag<HardEvent::MTE3_MTE1>(0);
    WaitFlag<HardEvent::MTE3_MTE1>(0);
    HighPrecLoad_32x32_B(l0b_temp0, l1_temp0[32 * 32 * 4]);
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    Mmad(l0c_temp0, l0a_temp0[32 * 32 * 3], l0b_temp0, mmadParams);
    SetFlag<HardEvent::M_V>(0);
    WaitFlag<HardEvent::M_V>(0);
    DataCopy(src[32 * 32 * 2], l0c_temp0, {1, 4, 0, 0}, enhancedParams);
    NZ2ND_32x32(dst[64 * 32], src[32 * 32 * 2]);
}

__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::Process() {
    LocalTensor<half> ub_temp_half_0 = ub_temp0.ReinterpretCast<half>();
    LocalTensor<half> ub_temp_half_1 = ub_temp1.ReinterpretCast<half>();
    LocalTensor<half> ub_temp_half_2 = ub_temp2.ReinterpretCast<half>();
    LocalTensor<half> ub_temp_half_4 = ub_temp4.ReinterpretCast<half>();
    LocalTensor<half> ub_temp_half_5 = ub_temp5.ReinterpretCast<half>();

    float q_scale = ((float)1.0) / __builtin_cce_sqrtf(headDimQKfp32);

    TransposeBetaHalf();
    PipeBarrier<PIPE_ALL>();
    TransposeGFloat();
    PipeBarrier<PIPE_ALL>();
    GCumsumFloat();
    PipeBarrier<PIPE_ALL>();
    Duplicate(I, (float)0, 16 * 16);
    PipeBarrier<PIPE_ALL>();
    for (int l = 0; l < 16; l++) {
        I[16 * l].SetValue(l, (float)1.0);
    }
    PipeBarrier<PIPE_ALL>();
    Muls(I2, I, (float)2, 16 * 16);
    Duplicate(ub_temp1[64 * 16], (float)1, 64);
    SetFlag<HardEvent::MTE3_V>(4);
    SetFlag<HardEvent::MTE3_MTE2>(4);
    SetFlag<HardEvent::MTE3_MTE2>(5);
    for (int bs = 0; bs < batchSize; bs++) {
        for (int nh = 0; nh < singleCoreNumHead; nh++) {
            int head_index = nh + block_id * singleCoreNumHead;
            for (int nc = 0; nc < (seqLenPaded / chunkSize); nc++) {
                // [BS, SEQ_LEN, NUM_HEAD, QK_HEAD_DIM/V_HEAD_DIM] ---> [BS, NUM_HEAD, SEQ_LEN (NUM_CHUNK, CHUNK_SIZE), QK_HEAD_DIM/V_HEAD_DIM]
                int index_base = (bs * numHead * seqLen + nc * numHead * chunkSize + head_index);
                int gbeta_index = bs * numHead * seqLenPaded + head_index * seqLenPaded + nc * chunkSize;
                int ls_index = (bs * numHead + head_index) * headDimV * headDimQK;
                DataCopy<half>(ub_temp1.ReinterpretCast<half>(), btransGlobal[gbeta_index], chunkSize);

                SetFlag<HardEvent::MTE2_V>(0);

                WaitFlag<HardEvent::MTE2_V>(0);
                Cast(ub_temp1[chunkSize], ub_temp1.ReinterpretCast<half>(), RoundMode::CAST_NONE, chunkSize);

                WaitFlag<HardEvent::MTE3_MTE2>(5);
                LoadQKVHalfWithTail(ub_temp_half_0, kGlobal[index_base * headDimQK], headDimQK, nc);
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
                DataCopy<float>(ub_temp1, gcumsumGlobal[gbeta_index], chunkSize);
                SetFlag<HardEvent::MTE2_V>(6);

                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                if ((nc == ((seqLenPaded / chunkSize) - 1)) && (tailSeqLen != 0)) {
                    Duplicate<float>(ub_temp12, (float)0, chunkSize * headDimQK);
                    L2normDim128FloatTrans(ub_temp_half_0, tailSeqLen);
                } else {
                    L2normDim128FloatTrans(ub_temp_half_0, chunkSize);
                }
                Cast(ub_temp_half_0, ub_temp6, RoundMode::CAST_ODD, chunkSize * headDimQK);
                WaitFlag<HardEvent::MTE3_MTE2>(4);
                WaitFlag<HardEvent::MTE3_V>(4);

                Mul(ub_temp4, ub_temp6, ub_temp1[chunkSize], 64, headDimQK, {1, 1, 1, 8, 8, 0});

                Cast(ub_temp2.ReinterpretCast<half>(), ub_temp4, RoundMode::CAST_ODD, chunkSize * headDimQK);

                UB2L1_ND2NZ<0>(l1_temp0, ub_temp2.ReinterpretCast<half>(), ub_temp12.ReinterpretCast<half>(), 128, 64);
                Duplicate<float>(ub_temp2, 0, chunkSize * chunkSize);
                SetFlag<HardEvent::MTE3_MTE1>(0);
                UB2L1_ND2NZ<1>(l1_temp0[64 * 128], ub_temp_half_0, ub_temp12.ReinterpretCast<half>()[64 * 128], 128, 64);
                {
                    TransDataTo5HDParams nchwconvParams;
                    nchwconvParams.repeatTimes = 4;
                    nchwconvParams.dstRepStride = 8;
                    nchwconvParams.srcRepStride = 1;

                    WaitFlag<HardEvent::MTE2_V>(6);
                    TransDataTo5HD<float>(outputPtr2, inputPtr2, nchwconvParams);
                    Sub(ub_temp4, ub_temp12, ub_temp1, 64, 64, {1, 0, 1, 8, 1, 0});
                    Exp<float>(ub_temp1, ub_temp1, chunkSize);
                    for (int l = 1; l < chunkSize; l++) {
                        Exp<float>(ub_temp2[l * chunkSize], ub_temp4[l * chunkSize], l);
                    }
                    
                }
                SetFlag<HardEvent::MTE3_MTE1>(1);
                WaitFlag<HardEvent::MTE3_MTE1>(0);
                MatMul<1, true, false>(ub_temp3, l1_temp0, l1_temp0[64 * 128], ub_temp12, 64, 64, 128, 0, 0, 128 * 128);
                
                SetFlag<HardEvent::V_S>(2);
                Mul<float>(ub_temp1[chunkSize * 2], ub_temp1, ub_temp1[chunkSize], chunkSize);
                Mul<float>(ub_temp3, ub_temp3, ub_temp2, chunkSize * chunkSize);
                
                LoadQKVHalfWithTail(ub_temp_half_5, qGlobal[index_base * headDimQK], headDimQK, nc);
                Inv_bisec16_opt<true>(ub_temp3, ub_temp3);

                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);

                if ((nc == ((seqLenPaded / chunkSize) - 1)) && (tailSeqLen != 0)) {
                    Duplicate<float>(ub_temp12, (float)0, chunkSize * headDimQK);
                    L2normDim128FloatTrans(ub_temp_half_5, tailSeqLen);
                } else {
                    L2normDim128FloatTrans(ub_temp_half_5, chunkSize);
                }

                Muls<float>(ub_temp6, ub_temp6, q_scale, chunkSize * headDimQK);
                Cast(ub_temp_half_5, ub_temp6, RoundMode::CAST_ODD, chunkSize * headDimQK);

                UB2L1_ND2NZ<0>(l1_temp0, ub_temp_half_5, ub_temp12.ReinterpretCast<half>(), 128, 64);
                SetFlag<HardEvent::MTE3_MTE1>(0);
                {
                    Mul<float>(ub_temp8, ub_temp3, ub_temp1[chunkSize * 2], 64, chunkSize, {1, 1, 1, 8, 8, 0});
                    Cast(ub_temp_half_5, ub_temp8, RoundMode::CAST_ODD, chunkSize * chunkSize);
                }
                SetFlag<HardEvent::MTE3_MTE1>(1);
                WaitFlag<HardEvent::MTE3_MTE1>(0);
                MatMul<1, true, false, true, false>(ub_temp10, l1_temp0, l1_temp0[64 * 128], ub_temp12, 64, 64, 128, 0, 0, 128 * 128);
                SetFlag<HardEvent::V_MTE2>(0);
                Mul<float>(ub_temp9, ub_temp10, ub_temp2, chunkSize * chunkSize);

                Cast(ub_temp_half_1[chunkSize * chunkSize], ub_temp9, RoundMode::CAST_ODD, chunkSize * chunkSize);
                WaitFlag<HardEvent::V_MTE2>(0);
                if (nc != 0) {
                    DataCopy(ub_temp11, lsGlobal[ls_index], headDimV * headDimQK);
                }

                Mul<float>(ub_temp6, ub_temp6, ub_temp1, 64, headDimQK, {1, 1, 1, 8, 8, 0});
                UB2L1_ND2NZ<0>(l1_temp0, ub_temp_half_5, ub_temp10.ReinterpretCast<half>(), 64, 64);
                SetFlag<HardEvent::MTE3_MTE1>(0);
                {
                    Mul<float>(ub_temp8, ub_temp3, ub_temp1[chunkSize], 64, chunkSize, {1, 1, 1, 8, 8, 0});
                    Cast(ub_temp_half_2, ub_temp8, RoundMode::CAST_ODD, chunkSize * chunkSize);
                    Cast(ub_temp_half_5, ub_temp6, RoundMode::CAST_ODD, chunkSize * headDimQK);
                    if (nc == 0) {
                        Duplicate<half>(ub_temp6.ReinterpretCast<half>(), (half)0, headDimV * headDimQK);
                    } else {
                        SetFlag<HardEvent::MTE2_V>(0);
                        WaitFlag<HardEvent::MTE2_V>(0);
                        Cast(ub_temp6.ReinterpretCast<half>(), ub_temp11, RoundMode::CAST_ODD, headDimV * headDimQK);
                    }
                }
                SetFlag<HardEvent::MTE3_MTE1>(1);
                WaitFlag<HardEvent::MTE3_MTE1>(0);
                MatMul<1, false, true, true, true>(ub_temp8, l1_temp0, l1_temp0[64 * 128], ub_temp10, 64, 128, 64, 0, 0, 128 * 128);
                UB2L1_ND2NZ<1>(l1_temp0[64 * 128 + 64 * 64 + 64 * 128], ub_temp6.ReinterpretCast<half>(), ub_temp12.ReinterpretCast<half>(), 128, 128);
                SetFlag<HardEvent::MTE3_MTE1>(1);
                Cast(ub_temp_half_0, ub_temp8, RoundMode::CAST_ODD, chunkSize * headDimQK);

                UB2L1_ND2NZ<0>(l1_temp0, ub_temp_half_0, ub_temp11.ReinterpretCast<half>(), 64, 128);
                {
                    SetFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::V_MTE2>(0);

                    LoadQKVHalfWithTail(ub_temp_half_0, vGlobal[index_base * headDimV], headDimV, nc);
                    SetFlag<HardEvent::MTE2_V>(1);
                }

                SetFlag<HardEvent::MTE3_MTE1>(0);

                WaitFlag<HardEvent::MTE3_MTE1>(0);
                MatMul<1, false, false>(ub_temp10, l1_temp0, l1_temp0[64 * 128 + 64 * 64 + 64 * 128], ub_temp12, 64, 128, 128, 0, 0, 128 * 128);

                WaitFlag<HardEvent::MTE2_V>(1);

                UB2L1_ND2NZ<7>(l1_temp0[128 * 128 * 4], ub_temp_half_2, ub_temp12.ReinterpretCast<half>(), 64, 64);
                SetFlag<HardEvent::MTE3_MTE1>(7);
                UB2L1_ND2NZ<1>(l1_temp0[64 * 64], ub_temp_half_0, ub_temp12.ReinterpretCast<half>()[64 * 64], 64, 128);
                SetFlag<HardEvent::MTE3_MTE1>(1);
                WaitFlag<HardEvent::MTE3_MTE1>(7);
                MatMul<1, false, false>(ub_temp11, l1_temp0[128 * 128 * 4], l1_temp0[64 * 64], ub_temp12, 64, 128, 64);

                Sub<float>(ub_temp12, ub_temp11, ub_temp10, chunkSize * headDimV);
                UB2L1_ND2NZ<0>(l1_temp0, ub_temp_half_5, ub_temp10.ReinterpretCast<half>(), 128, 64);
                SetFlag<HardEvent::MTE3_MTE1>(0);
                SetFlag<HardEvent::MTE3_MTE1>(1);
                Cast(ub_temp_half_2, ub_temp12, RoundMode::CAST_ODD, headDimV * chunkSize);
                {
                    UB2L1_ND2NZ<2>(l1_temp0[64 * 128], ub_temp_half_1[chunkSize * chunkSize], ub_temp10.ReinterpretCast<half>()[64 * 128], 64, 64);
                    SetFlag<HardEvent::MTE3_MTE1>(2);

                    UB2L1_ND2NZ<3>(l1_temp0[64 * 128 + 64 * 64], ub_temp_half_2, ub_temp11.ReinterpretCast<half>(), 64, 128);
                    SetFlag<HardEvent::MTE3_MTE1>(3);

                    WaitFlag<HardEvent::MTE3_MTE1>(0);
                    MatMul<1, true, false, true, false>(ub_temp5, l1_temp0, l1_temp0[64 * 128 + 64 * 64 + 64 * 128], ub_temp12, 64, 128, 128, 0, 0, 128 * 128);

                    SetFlag<HardEvent::M_MTE1>(0);
                    WaitFlag<HardEvent::M_MTE1>(0);

                    WaitFlag<HardEvent::MTE3_MTE1>(2);
                    MatMul<3, false, false>(ub_temp11, l1_temp0[64 * 128], l1_temp0[64 * 128 + 64 * 64], ub_temp12, 64, 128, 64, 64 * 128);
                }
                UB2L1_ND2NZ<0>(l1_temp0, ub_temp_half_4, ub_temp12.ReinterpretCast<half>(), 128, 64);
                SetFlag<HardEvent::MTE3_MTE1>(0);
                Add<float>(ub_temp5, ub_temp5, ub_temp11, chunkSize * headDimV);

                if (nc != 0) {
                    SetFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::V_MTE2>(0);
                    if (nc == 0) {
                        DataCopy(ub_temp9, isGlobal[ls_index], headDimV * headDimQK);
                    } else {
                        DataCopy(ub_temp9, lsGlobal[ls_index], headDimV * headDimQK);
                    }
                    SetFlag<HardEvent::MTE2_V>(0);
                }

                Cast(ub_temp_half_0, ub_temp5, RoundMode::CAST_ODD, headDimV * chunkSize);

                SetFlag<HardEvent::V_MTE3>(0);



                WaitFlag<HardEvent::V_MTE3>(0);
                StoreAttnHalfWithTail(attnGlobal[index_base * headDimV], ub_temp_half_0, headDimV, nc);
                SetFlag<HardEvent::MTE3_MTE2>(5);
                SetFlag<HardEvent::MTE3_MTE1>(1);
                WaitFlag<HardEvent::MTE3_MTE1>(0);
                if (nc == 0) {
                    MatMul<1, false, false, true, false>(ub_temp9, l1_temp0, l1_temp0[64 * 128 + 64 * 64], ub_temp12, 128, 128, 64); // ub_temp5 = (k_trans_i * decay_mask_i[-1]) @ v_new)
                } else {
                    MatMul<1, false, false, true, false>(ub_temp5, l1_temp0, l1_temp0[64 * 128 + 64 * 64], ub_temp12, 128, 128, 64); // ub_temp5 = (k_trans_i * decay_mask_i[-1]) @ v_new)
                }

                WaitFlag<HardEvent::V_S>(2);
                if (nc != 0) {

                    WaitFlag<HardEvent::MTE2_V>(0);

                    float g_exp_temp = ub_temp1.GetValue(63);
                    Muls<float>(ub_temp9, ub_temp9, g_exp_temp, headDimV * headDimQK);
                    Add<float>(ub_temp9, ub_temp9, ub_temp5, headDimV * headDimQK);
                }
                SetFlag<HardEvent::V_MTE3>(0);

                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopy<float>(lsGlobal[ls_index], ub_temp9, headDimQK * headDimV);

                SetFlag<HardEvent::MTE3_V>(4);
                SetFlag<HardEvent::MTE3_MTE2>(4);
            }
        }
    }
    WaitFlag<HardEvent::MTE3_V>(4);
    WaitFlag<HardEvent::MTE3_MTE2>(4);
    WaitFlag<HardEvent::MTE3_MTE2>(5);
}


__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::TransposeGFloat() {
    int blockLen = 16;
    int width = 8;
    int headGroupNum = numHead / width;
    int seqLenGroupNum = seqLenPaded / blockLen;
    int blockTailSeqLen = seqLen % blockLen;
    int fullBlockGroupNum = ceil_div(seqLen, blockLen);

    DataCopyParams repeatParams_in;
    repeatParams_in.blockLen = width * 4 / 32;
    repeatParams_in.srcGap = width * (headGroupNum - 1) * 4 / 32;
    repeatParams_in.dstGap = 0;
    repeatParams_in.blockCount = blockLen;

    DataCopyParams repeatParams_in_tail;
    repeatParams_in_tail.blockLen = width * 4 / 32;
    repeatParams_in_tail.srcGap = width * (headGroupNum - 1) * 4 / 32;
    repeatParams_in_tail.dstGap = 0;
    repeatParams_in_tail.blockCount = blockTailSeqLen;

    DataCopyParams repeatParams_out;
    repeatParams_out.blockLen = blockLen * 4 / 32;
    repeatParams_out.srcGap = 0;
    repeatParams_out.dstGap = blockLen * (seqLenGroupNum - 1) * 4 / 32;
    repeatParams_out.blockCount = width;

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = seqLenGroupNum;
    transDataParams.dstRepStride = blockLen;
    transDataParams.srcRepStride = blockLen;

    uint64_t dstLocalList[16];
    for (int b = 0; b < 16; b++) {
        dstLocalList[b] = (uint64_t)(ub_temp8[width * b].GetPhyAddr());
    }
    uint64_t srcLocalList[16];
    for (int b = 0; b < 16; b++) {
        srcLocalList[b] = (uint64_t)(ub_temp0[width * b].GetPhyAddr());
    }

    for (int i = 0; i < batchSize; i++) {
        for (int j_index = 0; j_index < headGroupNum; j_index++) {
            // UB tiling: ub_temp0/ub_temp8 each has 32768 floats => max repeats = 32768 / (width*blockLen) = 256
            const int maxSeqTile = 256;
            for (int k_base = 0; k_base < seqLenGroupNum; k_base += maxSeqTile) {
                int k_tile = seqLenGroupNum - k_base;
                if (k_tile > maxSeqTile) {
                    k_tile = maxSeqTile;
                }

                // Load a tile of [k_base, k_base + k_tile) blocks into ub_temp0
                for (int kk = 0; kk < k_tile; kk++) {
                    int k_index = k_base + kk;
                    if (k_index < fullBlockGroupNum) {
                        int src_index = i * numHead * seqLen + k_index * numHead * blockLen + j_index * width;
                        if (k_index == (fullBlockGroupNum - 1) && blockTailSeqLen != 0) {
                            Duplicate(ub_temp0[kk * width * blockLen], (float)0, width * blockLen);
                            SetFlag<HardEvent::V_MTE2>(0);
                            WaitFlag<HardEvent::V_MTE2>(0);
                            DataCopy<float>(ub_temp0[kk * width * blockLen], gGlobal[src_index], repeatParams_in_tail);
                        } else {
                            DataCopy<float>(ub_temp0[kk * width * blockLen], gGlobal[src_index], repeatParams_in);
                        }
                    } else {
                        Duplicate(ub_temp0[kk * width * blockLen], (float)0, width * blockLen);
                    }
                }

                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);

                TransDataTo5HDParams transDataParamsTile = transDataParams;
                transDataParamsTile.repeatTimes = k_tile;
                TransDataTo5HD<float>(dstLocalList, srcLocalList, transDataParamsTile);

                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);

                // Store the tile from ub_temp8 back to GM
                for (int kk = 0; kk < k_tile; kk++) {
                    int k_index = k_base + kk;
                    int dst_index = i * numHead * seqLenPaded + j_index * seqLenPaded * width + k_index * blockLen;
                    DataCopy<float>(gtransGlobal[dst_index], ub_temp8[kk * width * blockLen], repeatParams_out);
                }

                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
                SetFlag<HardEvent::MTE3_MTE2>(0);
                WaitFlag<HardEvent::MTE3_MTE2>(0);
            }
        }
    }
}


__aicore__ inline void ChunkGatedDeltaRuleV310Kernel::TransposeBetaHalf() {
    int blockLen = 16;
    int headGroupNum = numHead / blockLen;
    int seqLenGroupNum = seqLenPaded / blockLen;
    int blockTailSeqLen = seqLen % blockLen;
    int fullBlockGroupNum = ceil_div(seqLen, blockLen);

    DataCopyParams repeatParams_in;
    repeatParams_in.blockLen = blockLen * 2 / 32;
    repeatParams_in.srcGap = blockLen * (headGroupNum - 1) * 2 / 32;
    repeatParams_in.dstGap = 0;
    repeatParams_in.blockCount = blockLen;

    DataCopyParams repeatParams_in_tail;
    repeatParams_in_tail.blockLen = blockLen * 2 / 32;
    repeatParams_in_tail.srcGap = blockLen * (headGroupNum - 1) * 2 / 32;
    repeatParams_in_tail.dstGap = 0;
    repeatParams_in_tail.blockCount = blockTailSeqLen;

    DataCopyParams repeatParams_out;
    repeatParams_out.blockLen = blockLen * 2 / 32;
    repeatParams_out.srcGap = 0;
    repeatParams_out.dstGap = blockLen * (seqLenGroupNum - 1) * 2 / 32;
    repeatParams_out.blockCount = blockLen;

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = seqLenGroupNum * headGroupNum;
    transDataParams.dstRepStride = blockLen;
    transDataParams.srcRepStride = blockLen;

    LocalTensor<half> ub_temp_half_1 = ub_temp0.ReinterpretCast<half>();
    LocalTensor<half> ub_temp_half_2 = ub_temp8.ReinterpretCast<half>();

    for (int i = 0; i < batchSize; i++) {
        // Tile over seqLenGroupNum for each head group to avoid UB overflow.
        // One repeat is one 16x16 beta block (256 half). UB window (ub_temp_half_1/2) has 65536 half.
        // Max repeats per tile: 65536 / 256 = 256.
        const int maxSeqTile = 256;

        uint64_t dstLocalList[16];
        uint64_t srcLocalList[16];
        for (int b = 0; b < 16; b++) {
            srcLocalList[b] = (uint64_t)(ub_temp_half_1[blockLen * b].GetPhyAddr());
            dstLocalList[b] = (uint64_t)(ub_temp_half_2[blockLen * b].GetPhyAddr());
        }

        for (int j_index = 0; j_index < headGroupNum; j_index++) {
            for (int k_base = 0; k_base < seqLenGroupNum; k_base += maxSeqTile) {
                int k_tile = seqLenGroupNum - k_base;
                if (k_tile > maxSeqTile) {
                    k_tile = maxSeqTile;
                }

                // Load k_tile blocks for this head group into UB
                for (int kk = 0; kk < k_tile; kk++) {
                    int k_index = k_base + kk;
                    if (k_index < fullBlockGroupNum) {
                        int src_index = i * numHead * seqLen + k_index * numHead * blockLen + j_index * blockLen;
                        if (k_index == (fullBlockGroupNum - 1) && blockTailSeqLen != 0) {
                            Duplicate(ub_temp_half_1[kk * blockLen * blockLen], (half)0, blockLen * blockLen);
                            SetFlag<HardEvent::V_MTE2>(0);
                            WaitFlag<HardEvent::V_MTE2>(0);
                            DataCopy<half>(ub_temp_half_1[kk * blockLen * blockLen], bGlobal[src_index], repeatParams_in_tail);
                        } else {
                            DataCopy<half>(ub_temp_half_1[kk * blockLen * blockLen], bGlobal[src_index], repeatParams_in);
                        }
                    } else {
                        Duplicate(ub_temp_half_1[kk * blockLen * blockLen], (half)0, blockLen * blockLen);
                    }
                }

                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);

                TransDataTo5HDParams transDataParamsTile = transDataParams;
                transDataParamsTile.repeatTimes = k_tile;
                TransDataTo5HD<half>(dstLocalList, srcLocalList, transDataParamsTile);

                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);

                // Store k_tile blocks back to GM
                for (int kk = 0; kk < k_tile; kk++) {
                    int k_index = k_base + kk;
                    int dst_index = i * numHead * seqLenPaded + j_index * seqLenPaded * blockLen + k_index * blockLen;
                    DataCopy<half>(btransGlobal[dst_index], ub_temp_half_2[kk * blockLen * blockLen], repeatParams_out);
                }

                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
                SetFlag<HardEvent::MTE3_MTE2>(0);
                WaitFlag<HardEvent::MTE3_MTE2>(0);
            }
        }
    }
}

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_v310(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR g, GM_ADDR beta, GM_ADDR core_attn, GM_ADDR last_recurrent_state, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleTilingData);
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    ChunkGatedDeltaRuleV310Kernel kernel;
    kernel.Init(query, key, value, g, beta, core_attn, last_recurrent_state, workspace, tiling_data, &pipe);
    kernel.Process();
    PipeBarrier<PIPE_ALL>();
}
