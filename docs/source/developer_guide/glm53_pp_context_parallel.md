# GLM5.3 pipeline and DSA context parallelism with MLP offload

Apply the NZ prefetch loader and deferred shared-expert validation fixes before
deploying this combination. See [NZ offload](glm53_prefetch_nz_offload.md) and
[shared-expert execution](glm53_parallel_offload.md) for their separate contracts.

PP boundaries retain full-token tensors. Inside each PP stage, sequence-parallel
execution shards hidden states, residuals and deferred mHC post/comb state
together; outgoing tensors are gathered and padding is removed before the next
stage. Post/comb remain FP32, while positions and attention metadata describe the
full sequence. This introduces TP communication at non-final PP boundaries.

Only the first PP stage owns the input embedding. Context-parallel queries use
rank-local causal pool bounds, including empty and padded shards. The target
branch's native CANN FP8 indexer remains selected; INT8 and other supported
fallback paths retain their existing dispatch. The Triton KPool compressor avoids
reduction-based `tl.div_rn`, which fails in the installed Ascend compiler.

For the validated eager TP2/PP2 setup, retain FP8 KV/indexer caches, MTP5 and
MLP-only prefetch with group size 4, one offloaded layer per group and step 1:

```json
{
  "multistream_overlap_shared_expert": true,
  "enable_fused_mc2": 0,
  "enable_flashcomm1": true,
  "enable_dsa_cp": true,
  "c8_enable_reshape_optim": false
}
```

The focused CPU tests cover PP stages, causal metadata, cache construction and
FP32 mHC state. Full-service results are recorded in the associated PRs. Successful
one-token Prefill tests do not establish generated-text accuracy or validate a
complete P/D KV-transfer deployment. NZ prefetch graph capture remains outside
the validated eager configuration.
