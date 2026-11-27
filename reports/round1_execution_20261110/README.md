# reports/round1_execution_20261110

C5proxy, AdaBN, train_sweep_fasttrack_v5.py, and report code. The queues have been stopped.

[Reproduction guide](../../docs/REPRODUCING.md) · [Results](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

The parameters below were extracted statically from the code; they are not commands that automatically launch a run. For scripts without a CLI, check the constants, paths, and dependencies at the top of the file. Controllers and site runners require the original receipts and environments; do not bypass their guards.

| File | Declared CLI parameters |
|---|---|
| [adabn.py](adabn.py) | No argparse interface; read the source/config |
| [adabn_c5.py](adabn_c5.py) | No argparse interface; read the source/config |
| [aipex_data.py](aipex_data.py) | No argparse interface; read the source/config |
| [btc_c5ensemble.py](btc_c5ensemble.py) | No argparse interface; read the source/config |
| [c5prime.py](c5prime.py) | `--smoke` |
| [c5prime_recovery_v4.py](c5prime_recovery_v4.py) | `--smoke` |
| [c5prime_v2.py](c5prime_v2.py) | `--smoke` |
| [common.py](common.py) | No argparse interface; read the source/config |
| [complete_gop.py](complete_gop.py) | No argparse interface; read the source/config |
| [continue_af.py](continue_af.py) | No argparse interface; read the source/config |
| [continue_af_v2.py](continue_af_v2.py) | No argparse interface; read the source/config |
| [continue_af_v4.py](continue_af_v4.py) | No argparse interface; read the source/config |
| [continue_af_v5.py](continue_af_v5.py) | No argparse interface; read the source/config |
| [cpu_entry_v5.py](cpu_entry_v5.py) | No argparse interface; read the source/config |
| [d_sliding.py](d_sliding.py) | No argparse interface; read the source/config |
| [d_stack.py](d_stack.py) | No argparse interface; read the source/config |
| [deferred_eval_v5.py](deferred_eval_v5.py) | No argparse interface; read the source/config |
| [evaluate_c5.py](evaluate_c5.py) | No argparse interface; read the source/config |
| [fasttrack_gate_v5.py](fasttrack_gate_v5.py) | No argparse interface; read the source/config |
| [finalize_eligibility_v2.py](finalize_eligibility_v2.py) | No argparse interface; read the source/config |
| [inventory_plan.py](inventory_plan.py) | No argparse interface; read the source/config |
| [launch.py](launch.py) | No argparse interface; read the source/config |
| [launch_cpu.py](launch_cpu.py) | No argparse interface; read the source/config |
| [launch_postcheckpoint_v5.py](launch_postcheckpoint_v5.py) | No argparse interface; read the source/config |
| [make_plan.py](make_plan.py) | No argparse interface; read the source/config |
| [model_eval.py](model_eval.py) | No argparse interface; read the source/config |
| [octave_prepare.py](octave_prepare.py) | No argparse interface; read the source/config |
| [parallel_prepare.py](parallel_prepare.py) | No argparse interface; read the source/config |
| [portable_python.py](portable_python.py) | No argparse interface; read the source/config |
| [recover_c5_shard3_v4.py](recover_c5_shard3_v4.py) | No argparse interface; read the source/config |
| [refresh_a100_local.py](refresh_a100_local.py) | No argparse interface; read the source/config |
| [refresh_inventory.py](refresh_inventory.py) | No argparse interface; read the source/config |
| [report_boundaries.py](report_boundaries.py) | No argparse interface; read the source/config |
| [report_boundaries_v2.py](report_boundaries_v2.py) | No argparse interface; read the source/config |
| [report_boundaries_v4.py](report_boundaries_v4.py) | No argparse interface; read the source/config |
| [report_c5.py](report_c5.py) | No argparse interface; read the source/config |
| [report_cpu.py](report_cpu.py) | No argparse interface; read the source/config |
| [report_final.py](report_final.py) | No argparse interface; read the source/config |
| [report_metrics.py](report_metrics.py) | No argparse interface; read the source/config |
| [report_progress.py](report_progress.py) | No argparse interface; read the source/config |
| [revise_ieee_20261111.py](revise_ieee_20261111.py) | No argparse interface; read the source/config |
| [runtime_shared_v4.py](runtime_shared_v4.py) | No argparse interface; read the source/config |
| [select_d.py](select_d.py) | No argparse interface; read the source/config |
| [start_sweep.py](start_sweep.py) | No argparse interface; read the source/config |
| [start_sweep_eligible_v2.py](start_sweep_eligible_v2.py) | No argparse interface; read the source/config |
| [start_sweep_fasttrack_v5.py](start_sweep_fasttrack_v5.py) | No argparse interface; read the source/config |
| [start_sweep_shared_v4.py](start_sweep_shared_v4.py) | No argparse interface; read the source/config |
| [stop_superseded.py](stop_superseded.py) | No argparse interface; read the source/config |
| [switch_cpu_parallel.py](switch_cpu_parallel.py) | No argparse interface; read the source/config |
| [test_adabn.py](test_adabn.py) | No argparse interface; read the source/config |
| [test_fasttrack_gate_v5.py](test_fasttrack_gate_v5.py) | No argparse interface; read the source/config |
| [test_reporting.py](test_reporting.py) | No argparse interface; read the source/config |
| [test_runtime_shared_v4.py](test_runtime_shared_v4.py) | No argparse interface; read the source/config |
| [train_sweep.py](train_sweep.py) | No argparse interface; read the source/config |
| [train_sweep_eligible_v2.py](train_sweep_eligible_v2.py) | No argparse interface; read the source/config |
| [train_sweep_fasttrack_v5.py](train_sweep_fasttrack_v5.py) | No argparse interface; read the source/config |
| [training_provider.py](training_provider.py) | No argparse interface; read the source/config |
| [validate_sweep.py](validate_sweep.py) | No argparse interface; read the source/config |
| [verify_octave.py](verify_octave.py) | No argparse interface; read the source/config |
| [verify_octave_v2.py](verify_octave_v2.py) | No argparse interface; read the source/config |
| [wait_adabn.py](wait_adabn.py) | No argparse interface; read the source/config |
| [warm_constraints_v2.py](warm_constraints_v2.py) | No argparse interface; read the source/config |
| [warm_constraints_v3.py](warm_constraints_v3.py) | No argparse interface; read the source/config |
| [warm_training.py](warm_training.py) | No argparse interface; read the source/config |
