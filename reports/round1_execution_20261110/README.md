# reports/round1_execution_20261110

C5proxy, AdaBN, train_sweep_fasttrack_v5.py, report code. Queues đã dừng.

[Hướng dẫn chạy](../../docs/REPRODUCING.md) · [Kết quả](../../docs/RESULTS.md) · [Experiment map](../../docs/EXPERIMENTS.md)

Các tham số dưới đây trích tĩnh từ code, không phải lệnh tự động launch. Với script không có CLI, đọc hằng số/path và dependency ở đầu file. Controller/site runner cần receipt và môi trường gốc; không bỏ qua guard.

| File | Tham số CLI được khai báo |
|---|---|
| [adabn.py](adabn.py) | Không có argparse; đọc source/config |
| [adabn_c5.py](adabn_c5.py) | Không có argparse; đọc source/config |
| [aipex_data.py](aipex_data.py) | Không có argparse; đọc source/config |
| [btc_c5ensemble.py](btc_c5ensemble.py) | Không có argparse; đọc source/config |
| [c5prime.py](c5prime.py) | `--smoke` |
| [c5prime_recovery_v4.py](c5prime_recovery_v4.py) | `--smoke` |
| [c5prime_v2.py](c5prime_v2.py) | `--smoke` |
| [common.py](common.py) | Không có argparse; đọc source/config |
| [complete_gop.py](complete_gop.py) | Không có argparse; đọc source/config |
| [continue_af.py](continue_af.py) | Không có argparse; đọc source/config |
| [continue_af_v2.py](continue_af_v2.py) | Không có argparse; đọc source/config |
| [continue_af_v4.py](continue_af_v4.py) | Không có argparse; đọc source/config |
| [continue_af_v5.py](continue_af_v5.py) | Không có argparse; đọc source/config |
| [cpu_entry_v5.py](cpu_entry_v5.py) | Không có argparse; đọc source/config |
| [d_sliding.py](d_sliding.py) | Không có argparse; đọc source/config |
| [d_stack.py](d_stack.py) | Không có argparse; đọc source/config |
| [deferred_eval_v5.py](deferred_eval_v5.py) | Không có argparse; đọc source/config |
| [evaluate_c5.py](evaluate_c5.py) | Không có argparse; đọc source/config |
| [fasttrack_gate_v5.py](fasttrack_gate_v5.py) | Không có argparse; đọc source/config |
| [finalize_eligibility_v2.py](finalize_eligibility_v2.py) | Không có argparse; đọc source/config |
| [inventory_plan.py](inventory_plan.py) | Không có argparse; đọc source/config |
| [launch.py](launch.py) | Không có argparse; đọc source/config |
| [launch_cpu.py](launch_cpu.py) | Không có argparse; đọc source/config |
| [launch_postcheckpoint_v5.py](launch_postcheckpoint_v5.py) | Không có argparse; đọc source/config |
| [make_plan.py](make_plan.py) | Không có argparse; đọc source/config |
| [model_eval.py](model_eval.py) | Không có argparse; đọc source/config |
| [octave_prepare.py](octave_prepare.py) | Không có argparse; đọc source/config |
| [parallel_prepare.py](parallel_prepare.py) | Không có argparse; đọc source/config |
| [portable_python.py](portable_python.py) | Không có argparse; đọc source/config |
| [recover_c5_shard3_v4.py](recover_c5_shard3_v4.py) | Không có argparse; đọc source/config |
| [refresh_a100_local.py](refresh_a100_local.py) | Không có argparse; đọc source/config |
| [refresh_inventory.py](refresh_inventory.py) | Không có argparse; đọc source/config |
| [report_boundaries.py](report_boundaries.py) | Không có argparse; đọc source/config |
| [report_boundaries_v2.py](report_boundaries_v2.py) | Không có argparse; đọc source/config |
| [report_boundaries_v4.py](report_boundaries_v4.py) | Không có argparse; đọc source/config |
| [report_c5.py](report_c5.py) | Không có argparse; đọc source/config |
| [report_cpu.py](report_cpu.py) | Không có argparse; đọc source/config |
| [report_final.py](report_final.py) | Không có argparse; đọc source/config |
| [report_metrics.py](report_metrics.py) | Không có argparse; đọc source/config |
| [report_progress.py](report_progress.py) | Không có argparse; đọc source/config |
| [revise_ieee_20261111.py](revise_ieee_20261111.py) | Không có argparse; đọc source/config |
| [runtime_shared_v4.py](runtime_shared_v4.py) | Không có argparse; đọc source/config |
| [select_d.py](select_d.py) | Không có argparse; đọc source/config |
| [start_sweep.py](start_sweep.py) | Không có argparse; đọc source/config |
| [start_sweep_eligible_v2.py](start_sweep_eligible_v2.py) | Không có argparse; đọc source/config |
| [start_sweep_fasttrack_v5.py](start_sweep_fasttrack_v5.py) | Không có argparse; đọc source/config |
| [start_sweep_shared_v4.py](start_sweep_shared_v4.py) | Không có argparse; đọc source/config |
| [stop_superseded.py](stop_superseded.py) | Không có argparse; đọc source/config |
| [switch_cpu_parallel.py](switch_cpu_parallel.py) | Không có argparse; đọc source/config |
| [test_adabn.py](test_adabn.py) | Không có argparse; đọc source/config |
| [test_fasttrack_gate_v5.py](test_fasttrack_gate_v5.py) | Không có argparse; đọc source/config |
| [test_reporting.py](test_reporting.py) | Không có argparse; đọc source/config |
| [test_runtime_shared_v4.py](test_runtime_shared_v4.py) | Không có argparse; đọc source/config |
| [train_sweep.py](train_sweep.py) | Không có argparse; đọc source/config |
| [train_sweep_eligible_v2.py](train_sweep_eligible_v2.py) | Không có argparse; đọc source/config |
| [train_sweep_fasttrack_v5.py](train_sweep_fasttrack_v5.py) | Không có argparse; đọc source/config |
| [training_provider.py](training_provider.py) | Không có argparse; đọc source/config |
| [validate_sweep.py](validate_sweep.py) | Không có argparse; đọc source/config |
| [verify_octave.py](verify_octave.py) | Không có argparse; đọc source/config |
| [verify_octave_v2.py](verify_octave_v2.py) | Không có argparse; đọc source/config |
| [wait_adabn.py](wait_adabn.py) | Không có argparse; đọc source/config |
| [warm_constraints_v2.py](warm_constraints_v2.py) | Không có argparse; đọc source/config |
| [warm_constraints_v3.py](warm_constraints_v3.py) | Không có argparse; đọc source/config |
| [warm_training.py](warm_training.py) | Không có argparse; đọc source/config |
