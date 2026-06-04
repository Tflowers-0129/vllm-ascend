#include "kernel_operator.h"
#include "../op_host/chunk_gated_delta_rule_fwd_prepare_tiling.h"

using namespace AscendC;

namespace {

constexpr uint32_t CHUNK_SIZE = 64;
constexpr uint32_t MAX_K_DIM = 256;
constexpr uint32_t MAX_V_DIM = 256;
constexpr uint32_t ATTENTION_SIZE = CHUNK_SIZE * CHUNK_SIZE;
constexpr float L2NORM_EPS = 1e-6f;

template <typename T>
__aicore__ inline float ToFloatValue(T value)
{
    return static_cast<float>(value);
}

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

class ChunkGatedDeltaRuleFwdPrepareKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR q,
        GM_ADDR k,
        GM_ADDR v,
        GM_ADDR g,
        GM_ADDR beta,
        GM_ADDR qOut,
        GM_ADDR kOut,
        GM_ADDR wOut,
        GM_ADDR uOut,
        GM_ADDR gOut,
        GM_ADDR tiling)
    {
        auto *tilingData = reinterpret_cast<__gm__ optiling::ChunkGatedDeltaRuleFwdPrepareTilingData *>(tiling);
        batch_ = static_cast<uint32_t>(tilingData->batch);
        seqlen_ = static_cast<uint32_t>(tilingData->seqlen);
        kNumHead_ = static_cast<uint32_t>(tilingData->kNumHead);
        vNumHead_ = static_cast<uint32_t>(tilingData->vNumHead);
        kHeadDim_ = static_cast<uint32_t>(tilingData->kHeadDim);
        vHeadDim_ = static_cast<uint32_t>(tilingData->vHeadDim);
        chunkSize_ = static_cast<uint32_t>(tilingData->chunkSize);
        useQkL2norm_ = tilingData->useQkL2norm != 0;
        numChunks_ = seqlen_ / chunkSize_;
        headGroups_ = vNumHead_ / kNumHead_;

        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(v));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(g));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(beta));
        qOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(qOut));
        kOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(kOut));
        wOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(wOut));
        uOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(uOut));
        gOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gOut));

        pipe_.InitBuffer(gBuf_, AlignUp(CHUNK_SIZE * sizeof(float), 32));
        pipe_.InitBuffer(expGBuf_, AlignUp(CHUNK_SIZE * sizeof(float), 32));
        pipe_.InitBuffer(betaBuf_, AlignUp(CHUNK_SIZE * sizeof(float), 32));
        pipe_.InitBuffer(keyBuf_, AlignUp(CHUNK_SIZE * MAX_K_DIM * sizeof(float), 32));
        pipe_.InitBuffer(valueBuf_, AlignUp(CHUNK_SIZE * MAX_V_DIM * sizeof(float), 32));
        pipe_.InitBuffer(attnBuf_, AlignUp(ATTENTION_SIZE * sizeof(float), 32));
        pipe_.InitBuffer(rowBuf_, AlignUp(CHUNK_SIZE * sizeof(float), 32));
    }

    __aicore__ inline void Process()
    {
        TransposeQK();
        if (chunkSize_ != CHUNK_SIZE || kHeadDim_ > MAX_K_DIM || vHeadDim_ > MAX_V_DIM) {
            return;
        }
        const uint32_t taskCount = batch_ * vNumHead_ * numChunks_;
        const uint32_t coreNum = GetBlockNum();
        const uint32_t coreIdx = GetBlockIdx();
        for (uint32_t taskIdx = coreIdx; taskIdx < taskCount; taskIdx += coreNum) {
            ProcessOneChunk(taskIdx);
        }
    }

private:
    __aicore__ inline void TransposeQK()
    {
        const uint32_t coreNum = GetBlockNum();
        const uint32_t coreIdx = GetBlockIdx();
        const uint64_t totalRows = static_cast<uint64_t>(batch_) * kNumHead_ * seqlen_;
        for (uint64_t rowIdx = coreIdx; rowIdx < totalRows; rowIdx += coreNum) {
            uint64_t tmp = rowIdx;
            const uint32_t t = tmp % seqlen_;
            tmp /= seqlen_;
            const uint32_t h = tmp % kNumHead_;
            const uint32_t b = tmp / kNumHead_;
            const uint64_t inBase = ((static_cast<uint64_t>(b) * seqlen_ + t) * kNumHead_ + h) * kHeadDim_;
            const uint64_t outBase = rowIdx * kHeadDim_;

            float qScale = 1.0f;
            float kScale = 1.0f;
            if (useQkL2norm_) {
                float qSum = 0.0f;
                float kSum = 0.0f;
                for (uint32_t d = 0; d < kHeadDim_; ++d) {
                    const float qValue = ToFloatValue(qGm_.GetValue(inBase + d));
                    const float kValue = ToFloatValue(kGm_.GetValue(inBase + d));
                    qSum += qValue * qValue;
                    kSum += kValue * kValue;
                }
                qScale = 1.0f / sqrt(qSum + L2NORM_EPS);
                kScale = 1.0f / sqrt(kSum + L2NORM_EPS);
            }

            for (uint32_t d = 0; d < kHeadDim_; ++d) {
                const float qValue = ToFloatValue(qGm_.GetValue(inBase + d)) * qScale;
                const float kValue = ToFloatValue(kGm_.GetValue(inBase + d)) * kScale;
                qOutGm_.SetValue(outBase + d, static_cast<half>(qValue));
                kOutGm_.SetValue(outBase + d, static_cast<half>(kValue));
            }
        }
    }

    __aicore__ inline void ProcessOneChunk(uint32_t taskIdx)
    {
        const uint32_t chunkIdx = taskIdx % numChunks_;
        const uint32_t vHeadIdx = (taskIdx / numChunks_) % vNumHead_;
        const uint32_t batchIdx = taskIdx / (numChunks_ * vNumHead_);
        const uint32_t kHeadIdx = vHeadIdx / headGroups_;
        const uint32_t tokenBase = chunkIdx * CHUNK_SIZE;

        LocalTensor<float> gLocal = gBuf_.Get<float>();
        LocalTensor<float> expGLocal = expGBuf_.Get<float>();
        LocalTensor<float> betaLocal = betaBuf_.Get<float>();
        LocalTensor<float> keyLocal = keyBuf_.Get<float>();
        LocalTensor<float> valueLocal = valueBuf_.Get<float>();
        LocalTensor<float> attnLocal = attnBuf_.Get<float>();
        LocalTensor<float> rowLocal = rowBuf_.Get<float>();

        float gAcc = 0.0f;
        for (uint32_t i = 0; i < CHUNK_SIZE; ++i) {
            const uint32_t token = tokenBase + i;
            const uint64_t gateOffset = (static_cast<uint64_t>(batchIdx) * seqlen_ + token) * vNumHead_ + vHeadIdx;
            gAcc += gGm_.GetValue(gateOffset);
            gLocal.SetValue(i, gAcc);
            expGLocal.SetValue(i, gAcc);
            betaLocal.SetValue(i, ToFloatValue(betaGm_.GetValue(gateOffset)));
            const uint64_t gOutOffset = (static_cast<uint64_t>(batchIdx) * vNumHead_ + vHeadIdx) * seqlen_ + token;
            gOutGm_.SetValue(gOutOffset, gAcc);
        }
        Exp(expGLocal, expGLocal, CHUNK_SIZE);
        PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < CHUNK_SIZE; ++i) {
            const uint32_t token = tokenBase + i;
            float kScale = 1.0f;
            const uint64_t kBase = ((static_cast<uint64_t>(batchIdx) * seqlen_ + token) * kNumHead_ + kHeadIdx) *
                                   kHeadDim_;
            if (useQkL2norm_) {
                float kSum = 0.0f;
                for (uint32_t d = 0; d < kHeadDim_; ++d) {
                    const float kValue = ToFloatValue(kGm_.GetValue(kBase + d));
                    kSum += kValue * kValue;
                }
                kScale = 1.0f / sqrt(kSum + L2NORM_EPS);
            }
            for (uint32_t d = 0; d < kHeadDim_; ++d) {
                keyLocal.SetValue(i * kHeadDim_ + d, ToFloatValue(kGm_.GetValue(kBase + d)) * kScale);
            }
            for (uint32_t d = 0; d < vHeadDim_; ++d) {
                const uint64_t vOffset = ((static_cast<uint64_t>(batchIdx) * seqlen_ + token) * vNumHead_ + vHeadIdx) * vHeadDim_ + d;
                valueLocal.SetValue(i * vHeadDim_ + d, ToFloatValue(vGm_.GetValue(vOffset)));
            }
        }

        for (uint32_t i = 0; i < CHUNK_SIZE; ++i) {
            for (uint32_t j = 0; j < CHUNK_SIZE; ++j) {
                float value = 0.0f;
                if (j < i) {
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < kHeadDim_; ++d) {
                        dot += keyLocal.GetValue(i * kHeadDim_ + d) * keyLocal.GetValue(j * kHeadDim_ + d);
                    }
                    const float decay = expGLocal.GetValue(i) / expGLocal.GetValue(j);
                    value = -(dot * betaLocal.GetValue(i) * decay);
                }
                attnLocal.SetValue(i * CHUNK_SIZE + j, value);
            }
        }

        for (uint32_t i = 1; i < CHUNK_SIZE; ++i) {
            for (uint32_t j = 0; j < i; ++j) {
                rowLocal.SetValue(j, attnLocal.GetValue(i * CHUNK_SIZE + j));
            }
            for (uint32_t j = 0; j < i; ++j) {
                float value = rowLocal.GetValue(j);
                for (uint32_t l = 0; l < i; ++l) {
                    value += rowLocal.GetValue(l) * attnLocal.GetValue(l * CHUNK_SIZE + j);
                }
                attnLocal.SetValue(i * CHUNK_SIZE + j, value);
            }
        }
        for (uint32_t i = 0; i < CHUNK_SIZE; ++i) {
            attnLocal.SetValue(i * CHUNK_SIZE + i, 1.0f);
        }

        for (uint32_t i = 0; i < CHUNK_SIZE; ++i) {
            const uint32_t token = tokenBase + i;
            for (uint32_t d = 0; d < vHeadDim_; ++d) {
                float acc = 0.0f;
                for (uint32_t j = 0; j < CHUNK_SIZE; ++j) {
                    acc += attnLocal.GetValue(i * CHUNK_SIZE + j) * valueLocal.GetValue(j * vHeadDim_ + d) *
                           betaLocal.GetValue(j);
                }
                const uint64_t uOffset = (static_cast<uint64_t>(batchIdx) * vNumHead_ * seqlen_ +
                                          vHeadIdx * seqlen_ + token) *
                                             vHeadDim_ +
                                         d;
                uOutGm_.SetValue(uOffset, static_cast<half>(acc));
            }
            for (uint32_t d = 0; d < kHeadDim_; ++d) {
                float acc = 0.0f;
                for (uint32_t j = 0; j < CHUNK_SIZE; ++j) {
                    acc += attnLocal.GetValue(i * CHUNK_SIZE + j) * keyLocal.GetValue(j * kHeadDim_ + d) *
                           betaLocal.GetValue(j) * expGLocal.GetValue(j);
                }
                const uint64_t wOffset = (static_cast<uint64_t>(batchIdx) * vNumHead_ * seqlen_ +
                                          vHeadIdx * seqlen_ + token) *
                                             kHeadDim_ +
                                         d;
                wOutGm_.SetValue(wOffset, static_cast<half>(acc));
            }
        }
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> gBuf_;
    TBuf<QuePosition::VECCALC> expGBuf_;
    TBuf<QuePosition::VECCALC> betaBuf_;
    TBuf<QuePosition::VECCALC> keyBuf_;
    TBuf<QuePosition::VECCALC> valueBuf_;
    TBuf<QuePosition::VECCALC> attnBuf_;
    TBuf<QuePosition::VECCALC> rowBuf_;

    GlobalTensor<half> qGm_;
    GlobalTensor<half> kGm_;
    GlobalTensor<half> vGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<half> betaGm_;
    GlobalTensor<half> qOutGm_;
    GlobalTensor<half> kOutGm_;
    GlobalTensor<half> wOutGm_;
    GlobalTensor<half> uOutGm_;
    GlobalTensor<float> gOutGm_;

    uint32_t batch_;
    uint32_t seqlen_;
    uint32_t kNumHead_;
    uint32_t vNumHead_;
    uint32_t kHeadDim_;
    uint32_t vHeadDim_;
    uint32_t chunkSize_;
    uint32_t numChunks_;
    uint32_t headGroups_;
    bool useQkL2norm_;
};

} // namespace

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_fwd_prepare(
    GM_ADDR q,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR g,
    GM_ADDR beta,
    GM_ADDR qOut,
    GM_ADDR kOut,
    GM_ADDR wOut,
    GM_ADDR uOut,
    GM_ADDR gOut,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    ChunkGatedDeltaRuleFwdPrepareKernel op;
    op.Init(q, k, v, g, beta, qOut, kOut, wOut, uOut, gOut, tiling);
    op.Process();
}
