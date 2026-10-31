"""
run_matrix.py
=============
Sequential Execution Orchestrator for the CCA 1 30-Configuration Comparative Feature Selection Matrix.

Enforces:
1. Sequential execution with zero concurrency.
2. Complete process memory isolation for every configuration.
3. Strict pre-fit configuration-specific memory launch thresholds (SVM: 1536 MB, RF: 2560 MB).
4. Selector reuse: each selector is fitted once and reused across all 5 dimensions and both evaluators.
5. Bit-identical feature index slicing across classifiers at each dimension.
6. Verification of metrics and telemetry records after every single configuration.
7. Immediate sync of comparative_summary.json after every step.
8. Preservation of completed records (no silent overwriting).
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.feature_selection import ExperimentRunner

TRAIN_CACHE = Path(r"C:\Users\aryab\Coding\crop-disease-detection\data\cache\pv_train_subset_feats.npz")
VAL_CACHE = Path(r"C:\Users\aryab\Coding\crop-disease-detection\data\cache\pv_val_feats.npz")


def run_single(exp_id: str) -> None:
    """Worker function: executes a single configuration with clean memory isolation."""
    runner = ExperimentRunner()
    matrix = runner.get_experiment_matrix(include_rfe=False)
    exp_config = next((m for m in matrix if m["experiment_id"] == exp_id), None)
    if not exp_config:
        raise ValueError(f"Unknown experiment_id '{exp_id}'")

    print(f"[Worker] Executing configuration: {exp_id} ({exp_config['selector']} k={exp_config['k']} {exp_config['classifier']})")
    result = runner.run_single_configuration(exp_config, TRAIN_CACHE, VAL_CACHE, allow_retry=False)
    print(f"[Worker] Result status: {result['status']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCA 1 Feature Selection Matrix")
    parser.add_argument("--single", type=str, default=None, help="Execute a single configuration ID")
    args = parser.parse_args()

    if args.single:
        run_single(args.single)
        return

    runner = ExperimentRunner()
    matrix = runner.get_experiment_matrix(include_rfe=False)
    total_configs = len(matrix)

    print(f"================================================================================")
    print(f"CCA 1 COMPARATIVE MATRIX ORCHESTRATOR: {total_configs} CONFIGURATIONS")
    print(f"Train Cache: {TRAIN_CACHE}")
    print(f"Val Cache:   {VAL_CACHE}")
    print(f"================================================================================\n")

    completed_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, exp in enumerate(matrix, 1):
        exp_id = exp["experiment_id"]
        selector = exp["selector"]
        k = exp["k"]
        classifier = exp["classifier"]

        print(f"\n[{idx:02d}/{total_configs:02d}] Evaluating: {exp_id}")
        print(f"       Selector: {selector} | Dim: k={k} | Evaluator: {classifier}")

        # Check if already completed
        action, existing = runner.check_resumption(exp_id, allow_retry=False)
        if action == "skip":
            print(f"       Status: ALREADY COMPLETED (Skipping re-run in accordance with safety contract)")
            runner.verify_configuration_record(exp_id)
            skipped_count += 1
            completed_count += 1
            continue

        # Execute in an isolated subprocess to guarantee pristine OS-level memory availability
        cmd = [sys.executable, "-m", "src.feature_selection.run_matrix", "--single", exp_id]
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)

        if proc.returncode != 0:
            print(f"       [ERROR] Configuration {exp_id} failed with exit code {proc.returncode}!")
            print(f"       Worker STDERR:\n{proc.stderr}")
            failed_count += 1
            print(f"\n[ORCHESTRATOR HALT] Safeguard triggered. Halting execution immediately.")
            runner.sync_comparative_summary()
            sys.exit(1)

        # Output worker log highlights
        for line in proc.stdout.splitlines():
            if "[Worker]" in line:
                print(f"       {line}")

        # REQUIREMENT 10: Verify metrics and telemetry records immediately after completion
        try:
            runner.verify_configuration_record(exp_id)
            print(f"       [VERIFIED] Metrics (22 fields) and telemetry envelope strictly validated.")
            completed_count += 1
        except Exception as e:
            print(f"       [VERIFICATION FAILURE] Record integrity check failed for {exp_id}: {e}")
            runner.sync_comparative_summary()
            sys.exit(1)

        # Update master comparative summary after each configuration
        summary_path = runner.sync_comparative_summary()
        print(f"       [SYNC] comparative_summary.json updated ({completed_count} total records).")

    print(f"\n================================================================================")
    print(f"ALL {total_configs} CONFIGURATIONS PROCESSED SUCCESSFULLY!")
    print(f"Total: {total_configs} | Completed: {completed_count} (New: {completed_count - skipped_count}, Skipped Pilot: {skipped_count}) | Failed: {failed_count}")
    print(f"Master Summary: {runner.output_dir / 'comparative_summary.json'}")
    print(f"================================================================================")


if __name__ == "__main__":
    main()

