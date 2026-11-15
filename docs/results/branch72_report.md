> Historical 5 November aggregate report, all three seeds where stated. Not the later single-seed padding-fixed comparison. Candidate/status statements are superseded by the root README (15 November). Private receipts/checkpoints named below are not bundled.

# 72x72 branch: NOT RUN — storage guard

The revised request authorizes this experiment, superseding the earlier hold. Preflight ran while task1 was active. No cache, checkpoint or GPU training was launched. Completed runs0/15; fold-complete seeds0/3. No scientific gate can be evaluated.

WholeT8 was 63,149,660,677bytes, cap65,000,000,000bytes and reservedheadroom100,000,000bytes. Available 1,750,339,323bytes. The same frozen float16 layout at72 needs **140,512,662,144 additional RGBbytes**, before metadata/checkpoints. This comes from all1540 existing devcache headers. Raw datasets and existing outputs were preserved. Upscaling36 or substituting T3 affine was not used. Approval for an experiment did not waive the earlier storagecap. A new storage allocation or approved equivalent streaming design remains necessary.

| Input | Parameters | CPU median ms/10s clip |
|---|---:|---:|
| 36x36 | 467,337 | 685.391 |
| 72x72 | 2,163,081 | 4479.025 |

Benchmark: one CPU thread,2 warmups,5 forward passes, synthetic inputs and untrainedweights, excludes ROI/decoder/I/O. Parameter count changes because the existing architecture's dense head depends on image size; architecture definitions are otherwise unchanged. This is not a trained72 quality result.

The18November23:59 Asia/Saigon cutoff remains. No partial-fold score is substituted. Optional v3_joint_mcdnative was skipped because72 is not running and its prerequisites are unmet; it must not consume resources intended for72.
