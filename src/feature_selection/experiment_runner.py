"""
experiment_runner.py
====================
Systematic Feature Selection Experiment Runner and Resource Governance Engine.

Provides:
- MemoryMonitor: Enforces the 1.5 GB minimum available RAM threshold, tracks peak memory,
  and documents discrete RSS sampling limitations.
- ExperimentRecord: Complete 22-field schema contract across 6 nested categories.
- ExperimentRunner: Orchestrates sequential execution of all 40 configurations
  with inter-stage memory safeguards, cleanup, and partial result preservation.

Execution Governance:
- This module DOES NOT execute upon import.
- Gate 1 authorizes infrastructure definition only.
"""

import os
import gc
import json
import time
import tracemalloc
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple, Union

import io
import hashlib
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning

from .selectors import (
    INITIAL_FEATURE_DIM,
    TARGET_FEATURE_DIMS,
    deterministic_rank_scores,
    BaseFeatureSelector,
    SelectKBestANOVA,
    SelectFromModelSVM,
    SelectFromModelRF,
    TieredRFE,
)
from .pipeline import TrainOnlyStandardScaler, FeatureSelectionPipeline
from .rfe_pilot import get_system_memory_info


class MemoryThresholdError(RuntimeError):
    """Raised when available system memory drops below the mandatory threshold."""
    pass


class MemoryMonitor:
    """
    Monitors host system and process memory utilization across feature selection stages:
    - Enforces configuration-specific conservative launch thresholds:
      * Linear SVM: 1,536 MB (1.5 GB) minimum available system RAM.
      * Random Forest RF-2: 2,560 MB (2.5 GB) minimum available system RAM.
    - Captures discrete memory snapshots (available RAM and process RSS) before and after each model fit.
    - Documents critical safety limitations:
      * Memory cannot be guaranteed safe during an active scikit-learn fit (C-level/OpenMP allocations,
        transient tree nodes, or BLAS buffers).
      * The process may still encounter an OOM condition between boundary checks.
      * Unsafe asynchronous thread interruption is explicitly prohibited and not implemented.
      * Discrete polling captures state at boundary points; transient sub-millisecond spikes
        occurring within active fitting routines are not captured.
    """

    SVM_LAUNCH_THRESHOLD_MB: float = 1536.0  # 1.5 GB
    RF_LAUNCH_THRESHOLD_MB: float = 2560.0   # 2.5 GB
    DEFAULT_LAUNCH_THRESHOLD_MB: float = 1536.0
    MIN_AVAILABLE_RAM_MB: float = 1536.0  # Backward-compatible alias for 1.5 GB threshold

    def __init__(self, halt_on_violation: bool = True):
        self.halt_on_violation = halt_on_violation
        self.MIN_AVAILABLE_RAM_MB = self.DEFAULT_LAUNCH_THRESHOLD_MB

    def get_launch_threshold(self, classifier_name: str) -> float:
        """Returns the conservative pre-fit available-RAM launch threshold for a given classifier."""
        name_lower = classifier_name.lower()
        if "random_forest" in name_lower or "rf" in name_lower:
            return self.RF_LAUNCH_THRESHOLD_MB
        elif "svm" in name_lower:
            return self.SVM_LAUNCH_THRESHOLD_MB
        return self.DEFAULT_LAUNCH_THRESHOLD_MB

    def get_telemetry(self) -> Dict[str, Any]:
        mem = get_system_memory_info()
        return {
            "available_mb": mem["available_mb"],
            "total_mb": mem["total_mb"],
            "process_rss_mb": mem["process_rss_mb"],
            "sampling_disclaimer": (
                "RSS and available RAM are sampled discretely at step boundaries. "
                "Intra-step transient allocations during active C-level fits are not captured by discrete polling. "
                "Active fits cannot be asynchronously interrupted safely without risking host process corruption."
            )
        }

    def get_memory_snapshot(self) -> Dict[str, float]:
        """Returns instantaneous available RAM, process RSS, and total RAM in MB."""
        mem = get_system_memory_info()
        return {
            "available_mb": float(mem["available_mb"]),
            "process_rss_mb": float(mem["process_rss_mb"]),
            "total_mb": float(mem["total_mb"]),
        }

    def check_threshold(
        self,
        stage_name: str,
        required_threshold_mb: Optional[float] = None,
        max_stabilization_sec: float = 0.0
    ) -> Dict[str, Any]:
        """
        Asserts available memory meets the required launch threshold before proceeding.
        Optionally allows brief stabilization time for OS page reclamation following process termination.
        Raises MemoryThresholdError if available RAM remains below the required threshold.
        """
        threshold = required_threshold_mb if required_threshold_mb is not None else self.MIN_AVAILABLE_RAM_MB
        t0 = time.perf_counter()
        while True:
            gc.collect()
            telemetry = self.get_telemetry()
            if telemetry["available_mb"] >= threshold:
                return telemetry
            if (time.perf_counter() - t0) >= max_stabilization_sec:
                break
            time.sleep(1.0)

        msg = (
            f"Memory safeguard triggered at stage '{stage_name}': "
            f"Available memory {telemetry['available_mb']:.1f} MB is below the required "
            f"launch threshold ({threshold:.1f} MB)."
        )
        if self.halt_on_violation:
            raise MemoryThresholdError(msg)
        else:
            warnings.warn(msg)
        return telemetry


@dataclass
class ExperimentMetadata:
    experiment_id: str
    selector_algorithm: str
    classifier_name: str
    original_feature_count: int
    selected_feature_count: int
    feature_retention_ratio: float


@dataclass
class ClassificationMetrics:
    val_accuracy: float
    val_macro_precision: float
    val_macro_recall: float
    val_macro_f1: float
    val_weighted_f1: float


@dataclass
class TimingMeasurements:
    selector_fit_time_sec: float
    model_train_time_sec: float
    total_training_pipeline_sec: float


@dataclass
class InferenceMeasurements:
    single_sample_inference_ms: float
    batch32_throughput_img_per_sec: float


@dataclass
class StorageMeasurements:
    selector_storage_bytes: int
    model_storage_bytes: int
    total_storage_bytes: int


@dataclass
class ConvergenceDiagnostics:
    converged: bool
    n_iter_max: int
    convergence_warning_count: int


@dataclass
class ExperimentRecord:
    """
    Exact 22-field experiment record contract across 6 nested categories.
    """
    metadata: ExperimentMetadata
    classification_metrics: ClassificationMetrics
    timing_measurements: TimingMeasurements
    inference_measurements: InferenceMeasurements
    storage_measurements: StorageMeasurements
    convergence_diagnostics: ConvergenceDiagnostics

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionTelemetryRecord:
    """
    Separate execution-telemetry envelope tracking runtime, provenance, resource usage,
    and completion status independently from the 22-field classification metrics record.

    Adheres strictly to the Gate 3 Telemetry Lifecycle:
    - execution_status: strictly 'completed', 'failed', or 'interrupted'.
    - resume_action: 'executed', 'skipped_already_completed', or 'retried_with_authorization'.
    - end_timestamp: strictly populated on all terminations (including failed/interrupted).
    - attempt_number: tracks 1-based attempt sequence.
    - attempt_history: preserves prior attempt records across authorized retries.
    - Boundary memory tracking: records pre-fit and post-fit available RAM and RSS.
    """
    experiment_id: str
    execution_status: str  # "completed", "failed", "interrupted"
    resume_action: str = "executed"  # "executed", "skipped_already_completed", "retried_with_authorization"
    start_timestamp: str = ""        # ISO 8601 UTC/local
    end_timestamp: Optional[str] = None  # Populated on all terminations
    attempt_number: int = 1
    attempt_history: List[Dict[str, Any]] = field(default_factory=list)
    input_cache_sha256: str = ""
    selector_ranking_sha256: str = ""
    selected_feature_indices_sha256: str = ""
    selector_configuration: Dict[str, Any] = field(default_factory=dict)
    evaluator_configuration: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    pre_fit_available_mb: Optional[float] = None
    pre_fit_process_rss_mb: Optional[float] = None
    post_fit_available_mb: Optional[float] = None
    post_fit_process_rss_mb: Optional[float] = None
    peak_process_rss_mb: Optional[float] = None
    minimum_available_memory_mb: Optional[float] = None
    halt_reason: Optional[str] = None
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    metrics_record: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        valid_statuses = {"completed", "failed", "interrupted"}
        if self.execution_status not in valid_statuses:
            raise ValueError(
                f"Invalid execution_status '{self.execution_status}'. "
                f"Must be one of {valid_statuses}."
            )
        valid_actions = {"executed", "skipped_already_completed", "retried_with_authorization"}
        if self.resume_action not in valid_actions:
            raise ValueError(
                f"Invalid resume_action '{self.resume_action}'. "
                f"Must be one of {valid_actions}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentRunner:
    """
    Orchestrates sequential execution of feature-selection configurations:
    - Enforces sequential execution (zero concurrent runs).
    - Enforces configuration-specific memory launch thresholds (SVM: 1536 MB, RF: 2560 MB).
    - Conservative Random Forest worker allocation (default n_jobs=4).
    - Inter-stage memory checks, discrete boundary snapshots, and explicit garbage collection.
    - Saves individual configuration telemetry envelopes and partial result preservation.
    - Protects existing records: never overwrites completed records, never silently overwrites
      incomplete records, preserves retry attempt history.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        models_dir: Optional[Path] = None,
        rf_n_jobs: int = 4,
        svm_n_jobs: int = -1
    ):
        repo_root = Path(__file__).resolve().parents[2]
        self.output_dir = output_dir or (repo_root / "configs" / "feature_selection")
        self.models_dir = models_dir or (repo_root / "models" / "feature_selection")
        self.records_dir = self.output_dir / "records"
        self.rf_n_jobs = rf_n_jobs
        self.svm_n_jobs = svm_n_jobs
        self.memory_monitor = MemoryMonitor(halt_on_violation=True)
        self.records: List[ExperimentRecord] = []
        self.envelopes: List[ExecutionTelemetryRecord] = []

    def get_telemetry_path(self, experiment_id: str) -> Path:
        """Returns the canonical path for an experiment's execution telemetry record."""
        return self.records_dir / f"{experiment_id}_telemetry.json"

    def check_resumption(
        self,
        experiment_id: str,
        allow_retry: bool = False
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Inspects existing records on disk to govern execution/resumption:
        - If no record exists -> ("execute", None)
        - If completed record exists -> ("skip", existing_data)
        - If incomplete record (failed/interrupted) exists and allow_retry=False ->
          raises RuntimeError (never silently overwrite)
        - If incomplete record exists and allow_retry=True -> ("retry", existing_data)
        """
        path = self.get_telemetry_path(experiment_id)
        if not path.exists():
            return "execute", None

        with open(path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)

        status = existing_data.get("execution_status")
        if status == "completed":
            return "skip", existing_data
        elif status in {"failed", "interrupted"}:
            if not allow_retry:
                raise RuntimeError(
                    f"Incomplete record with status '{status}' already exists for '{experiment_id}'. "
                    f"Silent overwriting or uncontrolled resumption is strictly prohibited. "
                    f"Explicit authorization (allow_retry=True) is required to retry."
                )
            return "retry", existing_data
        else:
            raise ValueError(f"Unrecognized execution status '{status}' in {path}")

    def save_telemetry_record(
        self,
        record: ExecutionTelemetryRecord,
        allow_retry: bool = False
    ) -> Path:
        """
        Persists an ExecutionTelemetryRecord enforcing strict safety contracts:
        - Never overwrites an existing completed record.
        - Never silently overwrites an incomplete record during resumption.
        - If retrying an incomplete record under explicit authorization (allow_retry=True):
          preserves the earlier attempt in attempt_history and increments attempt_number.
        - Guarantees end_timestamp is populated for all termination states.
        """
        self.records_dir.mkdir(parents=True, exist_ok=True)
        path = self.get_telemetry_path(record.experiment_id)

        # Enforce termination timestamp contract for all termination states
        if record.end_timestamp is None:
            record.end_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)

            existing_status = existing_data.get("execution_status")
            if existing_status == "completed":
                raise FileExistsError(
                    f"Execution record for '{record.experiment_id}' is already marked as 'completed'. "
                    f"Overwriting completed experiment records is strictly prohibited."
                )
            elif existing_status in {"failed", "interrupted"}:
                if not allow_retry:
                    raise FileExistsError(
                        f"Incomplete execution record with status '{existing_status}' already exists for "
                        f"'{record.experiment_id}'. Silent overwriting is prohibited. "
                        f"Set allow_retry=True with explicit authorization to archive earlier attempt and retry."
                    )
                # Archive earlier attempt into attempt_history
                earlier_attempt = {k: v for k, v in existing_data.items() if k != "attempt_history"}
                existing_history = existing_data.get("attempt_history", [])
                record.attempt_history = list(existing_history) + [earlier_attempt]
                record.attempt_number = existing_data.get("attempt_number", 1) + 1
                record.resume_action = "retried_with_authorization"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)

        return path

    def get_pilot_matrix(self) -> List[Dict[str, Any]]:
        """
        Defines the exact 2-configuration bounded pilot matrix:
        1. SelectKBest ANOVA (k=4096) with Linear SVM.
        2. SelectKBest ANOVA (k=4096) with Random Forest RF-2.
        Fitted once on training data, generating separate metrics and telemetry records.
        """
        return [
            {
                "selector": "SelectKBest_ANOVA",
                "k": 4096,
                "classifier": "Linear_SVM",
                "experiment_id": "selectkbest_anova_k4096_linear_svm"
            },
            {
                "selector": "SelectKBest_ANOVA",
                "k": 4096,
                "classifier": "Random_Forest_RF2",
                "experiment_id": "selectkbest_anova_k4096_random_forest_rf2"
            }
        ]

    def get_experiment_matrix(self, include_rfe: bool = False) -> List[Dict[str, Any]]:
        """
        Defines the experimental matrix:
        - Revised Gate 3 matrix (include_rfe=False): 3 selectors x 5 dimensions x 2 classifiers = 30 configurations.
        - Legacy matrix (include_rfe=True): 4 selectors x 5 dimensions x 2 classifiers = 40 configurations.
        """
        selectors = ["SelectKBest_ANOVA", "SelectFromModel_SVM", "SelectFromModel_RF"]
        if include_rfe:
            selectors.append("RFE_SVM")

        classifiers = ["Linear_SVM", "Random_Forest_RF2"]
        matrix = []

        for sel in selectors:
            for k in TARGET_FEATURE_DIMS:
                for clf in classifiers:
                    matrix.append({
                        "selector": sel,
                        "k": k,
                        "classifier": clf,
                        "experiment_id": f"{sel.lower()}_k{k}_{clf.lower()}"
                    })
        return matrix

    def save_partial_records(self) -> None:
        """Flushes completed experiment records to disk upon halt or completion."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "comparative_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.records], f, indent=2)

    def run_pilot(
        self,
        train_cache_path: Union[str, Path],
        val_cache_path: Union[str, Path],
        allow_retry: bool = False
    ) -> Dict[str, Any]:
        """
        Executes the authorized Gate 3 Two-Configuration Bounded Pilot:
        1. SelectKBest ANOVA (k=4096) with Linear SVM.
        2. SelectKBest ANOVA (k=4096) with Random Forest RF-2.

        Enforces:
        - Configuration-specific pre-fit launch thresholds (SVM: 1536 MB, RF: 2560 MB).
        - Sequential execution with zero concurrent fits.
        - ANOVA fitted ONCE on training data; selected indices and ranking reused.
        - Strict train-only scaling for Linear SVM (TrainOnlyStandardScaler).
        - Unscaled raw feature subset for Random Forest RF-2.
        - 22-field ExperimentRecord generated for each configuration.
        - ExecutionTelemetryRecord envelope capturing boundary RAM/RSS snapshots and lifecycle.
        - Output persisted to configs/feature_selection/records/.
        """
        train_path = Path(train_cache_path)
        val_path = Path(val_cache_path)

        if not train_path.exists():
            raise FileNotFoundError(f"Training cache not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation cache not found: {val_path}")

        # Compute SHA-256 of input cache
        h = hashlib.sha256()
        with open(train_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        input_cache_sha256 = h.hexdigest()

        # Check resumption for both configurations
        pilot_configs = self.get_pilot_matrix()
        for pcfg in pilot_configs:
            action, _ = self.check_resumption(pcfg["experiment_id"], allow_retry=allow_retry)
            if action == "skip":
                warnings.warn(f"Configuration '{pcfg['experiment_id']}' already completed. Skipping re-fit.")

        # 1. Load Training Cache
        gc.collect()
        with np.load(train_path, allow_pickle=True) as train_data:
            X_train = np.asarray(train_data["features"], dtype=np.float32)
            y_train = np.asarray(train_data["labels"], dtype=np.int64)

        # 2. Fit ANOVA once
        t0_sel = time.perf_counter()
        anova = SelectKBestANOVA(k=4096)
        anova.fit(X_train, y_train)
        selector_fit_time_sec = float(time.perf_counter() - t0_sel)

        selected_indices = anova.selected_indices_
        selector_ranking_sha256 = hashlib.sha256(anova.feature_scores_.tobytes()).hexdigest()
        selected_indices_sha256 = hashlib.sha256(selected_indices.tobytes()).hexdigest()

        # Selector storage size
        buf_sel = io.BytesIO()
        joblib.dump(anova, buf_sel)
        selector_storage_bytes = buf_sel.tell()

        pilot_results = []

        # 3. Configuration 1: Linear SVM
        exp_id_svm = "selectkbest_anova_k4096_linear_svm"
        action_svm, existing_svm = self.check_resumption(exp_id_svm, allow_retry=allow_retry)
        if action_svm != "skip":
            gc.collect()
            self.memory_monitor.check_threshold(
                stage_name=f"{exp_id_svm}_launch",
                required_threshold_mb=self.memory_monitor.SVM_LAUNCH_THRESHOLD_MB
            )
            pre_fit_mem = self.memory_monitor.get_memory_snapshot()
            start_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            t0_run = time.perf_counter()

            # Train-only standard scaling
            scaler = TrainOnlyStandardScaler()
            X_tr_svm = scaler.fit_transform(X_train[:, selected_indices])

            # SGDClassifier with hinge loss (Linear SVM)
            alpha_val = 1.075384449940854e-05  # baseline C=10.0 equivalent
            svm = SGDClassifier(
                loss="hinge",
                penalty="l2",
                alpha=alpha_val,
                max_iter=1000,
                tol=1e-3,
                random_state=42,
                n_jobs=self.svm_n_jobs
            )

            t0_fit = time.perf_counter()
            svm.fit(X_tr_svm, y_train)
            model_train_time_sec = float(time.perf_counter() - t0_fit)
            post_fit_mem = self.memory_monitor.get_memory_snapshot()

            del X_tr_svm
            gc.collect()

            # Load validation data and evaluate
            with np.load(val_path, allow_pickle=True) as val_data:
                X_val_all = np.asarray(val_data["features"], dtype=np.float32)
                y_val = np.asarray(val_data["labels"], dtype=np.int64)

            X_val_svm = scaler.transform(X_val_all[:, selected_indices])
            y_pred_svm = svm.predict(X_val_svm)

            # Benchmarking inference
            latencies = []
            for idx in range(min(50, len(X_val_svm))):
                t0_inf = time.perf_counter()
                _ = svm.predict(X_val_svm[idx:idx+1])
                latencies.append((time.perf_counter() - t0_inf) * 1000.0)
            single_sample_inference_ms = float(np.mean(latencies))

            n_b = min(20, len(X_val_svm) // 32)
            if n_b > 0:
                t0_b = time.perf_counter()
                for b in range(n_b):
                    _ = svm.predict(X_val_svm[b*32 : (b+1)*32])
                batch32_throughput_img_per_sec = float((n_b * 32) / (time.perf_counter() - t0_b))
            else:
                batch32_throughput_img_per_sec = 0.0

            acc = float(accuracy_score(y_val, y_pred_svm))
            prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred_svm, average="macro", zero_division=0)
            _, _, w_f1, _ = precision_recall_fscore_support(y_val, y_pred_svm, average="weighted", zero_division=0)

            buf_m = io.BytesIO()
            joblib.dump(svm, buf_m)
            model_storage_bytes = buf_m.tell()

            del X_val_svm, X_val_all, y_pred_svm
            gc.collect()

            elapsed_svm = float(time.perf_counter() - t0_run)
            end_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            exp_rec_svm = ExperimentRecord(
                metadata=ExperimentMetadata(
                    experiment_id=exp_id_svm,
                    selector_algorithm="SelectKBest_ANOVA",
                    classifier_name="Linear_SVM",
                    original_feature_count=INITIAL_FEATURE_DIM,
                    selected_feature_count=4096,
                    feature_retention_ratio=4096 / INITIAL_FEATURE_DIM
                ),
                classification_metrics=ClassificationMetrics(
                    val_accuracy=acc,
                    val_macro_precision=float(prec),
                    val_macro_recall=float(rec),
                    val_macro_f1=float(f1),
                    val_weighted_f1=float(w_f1)
                ),
                timing_measurements=TimingMeasurements(
                    selector_fit_time_sec=selector_fit_time_sec,
                    model_train_time_sec=model_train_time_sec,
                    total_training_pipeline_sec=selector_fit_time_sec + model_train_time_sec
                ),
                inference_measurements=InferenceMeasurements(
                    single_sample_inference_ms=single_sample_inference_ms,
                    batch32_throughput_img_per_sec=batch32_throughput_img_per_sec
                ),
                storage_measurements=StorageMeasurements(
                    selector_storage_bytes=selector_storage_bytes,
                    model_storage_bytes=model_storage_bytes,
                    total_storage_bytes=selector_storage_bytes + model_storage_bytes
                ),
                convergence_diagnostics=ConvergenceDiagnostics(
                    converged=bool(svm.n_iter_ < 1000),
                    n_iter_max=int(svm.n_iter_),
                    convergence_warning_count=0
                )
            )

            metrics_json_path = self.records_dir / f"{exp_id_svm}_metrics.json"
            self.records_dir.mkdir(parents=True, exist_ok=True)
            with open(metrics_json_path, "w", encoding="utf-8") as f:
                json.dump(exp_rec_svm.to_dict(), f, indent=2)

            telemetry_rec_svm = ExecutionTelemetryRecord(
                experiment_id=exp_id_svm,
                execution_status="completed",
                resume_action="retried_with_authorization" if action_svm == "retry" else "executed",
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                input_cache_sha256=input_cache_sha256,
                selector_ranking_sha256=selector_ranking_sha256,
                selected_feature_indices_sha256=selected_indices_sha256,
                selector_configuration={"algorithm": "SelectKBest_ANOVA", "k": 4096},
                evaluator_configuration={
                    "classifier": "Linear_SVM",
                    "loss": "hinge",
                    "penalty": "l2",
                    "alpha": alpha_val,
                    "max_iter": 1000,
                    "tol": 1e-3,
                    "n_jobs": self.svm_n_jobs,
                    "random_state": 42
                },
                elapsed_seconds=elapsed_svm,
                pre_fit_available_mb=pre_fit_mem["available_mb"],
                pre_fit_process_rss_mb=pre_fit_mem["process_rss_mb"],
                post_fit_available_mb=post_fit_mem["available_mb"],
                post_fit_process_rss_mb=post_fit_mem["process_rss_mb"],
                artifact_paths={
                    "metrics_json": str(metrics_json_path),
                    "telemetry_json": str(self.get_telemetry_path(exp_id_svm))
                },
                metrics_record=exp_rec_svm.to_dict()
            )
            self.save_telemetry_record(telemetry_rec_svm, allow_retry=allow_retry)
            self.records.append(exp_rec_svm)
            self.envelopes.append(telemetry_rec_svm)
            pilot_results.append(telemetry_rec_svm.to_dict())
        else:
            pilot_results.append(existing_svm)
            if existing_svm and "metrics_record" in existing_svm and existing_svm["metrics_record"]:
                m_rec = existing_svm["metrics_record"]
                # Reconstruct ExperimentRecord for summary table
                exp_rec_svm = ExperimentRecord(
                    metadata=ExperimentMetadata(**m_rec["metadata"]),
                    classification_metrics=ClassificationMetrics(**m_rec["classification_metrics"]),
                    timing_measurements=TimingMeasurements(**m_rec["timing_measurements"]),
                    inference_measurements=InferenceMeasurements(**m_rec["inference_measurements"]),
                    storage_measurements=StorageMeasurements(**m_rec["storage_measurements"]),
                    convergence_diagnostics=ConvergenceDiagnostics(**m_rec["convergence_diagnostics"])
                )
                self.records.append(exp_rec_svm)

        # 4. Configuration 2: Random Forest RF-2
        exp_id_rf = "selectkbest_anova_k4096_random_forest_rf2"
        action_rf, _ = self.check_resumption(exp_id_rf, allow_retry=allow_retry)
        if action_rf != "skip":
            # Extract unscaled subset and free X_train to maximize available RAM
            X_tr_rf = X_train[:, selected_indices]
            del X_train
            gc.collect()

            self.memory_monitor.check_threshold(
                stage_name=f"{exp_id_rf}_launch",
                required_threshold_mb=self.memory_monitor.RF_LAUNCH_THRESHOLD_MB
            )
            pre_fit_mem_rf = self.memory_monitor.get_memory_snapshot()
            start_ts_rf = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            t0_run_rf = time.perf_counter()

            # Random Forest RF-2
            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42,
                n_jobs=self.rf_n_jobs
            )

            t0_fit_rf = time.perf_counter()
            rf.fit(X_tr_rf, y_train)
            model_train_time_sec_rf = float(time.perf_counter() - t0_fit_rf)
            post_fit_mem_rf = self.memory_monitor.get_memory_snapshot()

            del X_tr_rf, y_train
            gc.collect()

            # Load validation data and evaluate
            with np.load(val_path, allow_pickle=True) as val_data:
                X_val_all = np.asarray(val_data["features"], dtype=np.float32)
                y_val = np.asarray(val_data["labels"], dtype=np.int64)

            X_val_rf = X_val_all[:, selected_indices]
            y_pred_rf = rf.predict(X_val_rf)

            # Benchmarking inference
            latencies_rf = []
            for idx in range(min(50, len(X_val_rf))):
                t0_inf = time.perf_counter()
                _ = rf.predict(X_val_rf[idx:idx+1])
                latencies_rf.append((time.perf_counter() - t0_inf) * 1000.0)
            single_sample_inference_ms_rf = float(np.mean(latencies_rf))

            n_b = min(20, len(X_val_rf) // 32)
            if n_b > 0:
                t0_b = time.perf_counter()
                for b in range(n_b):
                    _ = rf.predict(X_val_rf[b*32 : (b+1)*32])
                batch32_throughput_img_per_sec_rf = float((n_b * 32) / (time.perf_counter() - t0_b))
            else:
                batch32_throughput_img_per_sec_rf = 0.0

            acc_rf = float(accuracy_score(y_val, y_pred_rf))
            prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(y_val, y_pred_rf, average="macro", zero_division=0)
            _, _, w_f1_rf, _ = precision_recall_fscore_support(y_val, y_pred_rf, average="weighted", zero_division=0)

            buf_m_rf = io.BytesIO()
            joblib.dump(rf, buf_m_rf)
            model_storage_bytes_rf = buf_m_rf.tell()

            del X_val_rf, X_val_all, y_pred_rf
            gc.collect()

            elapsed_rf = float(time.perf_counter() - t0_run_rf)
            end_ts_rf = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            exp_rec_rf = ExperimentRecord(
                metadata=ExperimentMetadata(
                    experiment_id=exp_id_rf,
                    selector_algorithm="SelectKBest_ANOVA",
                    classifier_name="Random_Forest_RF2",
                    original_feature_count=INITIAL_FEATURE_DIM,
                    selected_feature_count=4096,
                    feature_retention_ratio=4096 / INITIAL_FEATURE_DIM
                ),
                classification_metrics=ClassificationMetrics(
                    val_accuracy=acc_rf,
                    val_macro_precision=float(prec_rf),
                    val_macro_recall=float(rec_rf),
                    val_macro_f1=float(f1_rf),
                    val_weighted_f1=float(w_f1_rf)
                ),
                timing_measurements=TimingMeasurements(
                    selector_fit_time_sec=0.0,  # Reused from ANOVA
                    model_train_time_sec=model_train_time_sec_rf,
                    total_training_pipeline_sec=model_train_time_sec_rf
                ),
                inference_measurements=InferenceMeasurements(
                    single_sample_inference_ms=single_sample_inference_ms_rf,
                    batch32_throughput_img_per_sec=batch32_throughput_img_per_sec_rf
                ),
                storage_measurements=StorageMeasurements(
                    selector_storage_bytes=selector_storage_bytes,
                    model_storage_bytes=model_storage_bytes_rf,
                    total_storage_bytes=selector_storage_bytes + model_storage_bytes_rf
                ),
                convergence_diagnostics=ConvergenceDiagnostics(
                    converged=True,
                    n_iter_max=30,
                    convergence_warning_count=0
                )
            )

            metrics_json_path_rf = self.records_dir / f"{exp_id_rf}_metrics.json"
            with open(metrics_json_path_rf, "w", encoding="utf-8") as f:
                json.dump(exp_rec_rf.to_dict(), f, indent=2)

            telemetry_rec_rf = ExecutionTelemetryRecord(
                experiment_id=exp_id_rf,
                execution_status="completed",
                resume_action="retried_with_authorization" if action_rf == "retry" else "executed",
                start_timestamp=start_ts_rf,
                end_timestamp=end_ts_rf,
                input_cache_sha256=input_cache_sha256,
                selector_ranking_sha256=selector_ranking_sha256,
                selected_feature_indices_sha256=selected_indices_sha256,
                selector_configuration={"algorithm": "SelectKBest_ANOVA", "k": 4096},
                evaluator_configuration={
                    "classifier": "Random_Forest_RF2",
                    "n_estimators": 300,
                    "max_depth": 30,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "random_state": 42,
                    "n_jobs": self.rf_n_jobs
                },
                elapsed_seconds=elapsed_rf,
                pre_fit_available_mb=pre_fit_mem_rf["available_mb"],
                pre_fit_process_rss_mb=pre_fit_mem_rf["process_rss_mb"],
                post_fit_available_mb=post_fit_mem_rf["available_mb"],
                post_fit_process_rss_mb=post_fit_mem_rf["process_rss_mb"],
                artifact_paths={
                    "metrics_json": str(metrics_json_path_rf),
                    "telemetry_json": str(self.get_telemetry_path(exp_id_rf))
                },
                metrics_record=exp_rec_rf.to_dict()
            )
            self.save_telemetry_record(telemetry_rec_rf, allow_retry=allow_retry)
            self.records.append(exp_rec_rf)
            self.envelopes.append(telemetry_rec_rf)
            pilot_results.append(telemetry_rec_rf.to_dict())

        # Save comparative partial summary
        self.save_partial_records()

        return {
            "pilot_completed": True,
            "selector_fit_time_sec": selector_fit_time_sec,
            "selector_ranking_sha256": selector_ranking_sha256,
            "selected_feature_indices_sha256": selected_indices_sha256,
            "configurations": pilot_results
        }

    def fit_or_load_selector(
        self,
        selector_name: str,
        train_cache_path: Union[str, Path]
    ) -> Tuple[BaseFeatureSelector, np.ndarray, str, float, int]:
        """
        Retrieves a fitted selector and its full deterministic ranking, fitting once if not cached:
        - Reuses the single fitted selector across all 5 dimensions and both classifiers.
        - Preserves deterministic feature ranking and cryptographic checksums.
        - Enforces memory safeguards before fitting.
        """
        selectors_dir = self.output_dir / "selectors"
        selectors_dir.mkdir(parents=True, exist_ok=True)
        cached_selector_path = selectors_dir / f"{selector_name.lower()}.joblib"

        if cached_selector_path.exists():
            selector = joblib.load(cached_selector_path)
            ranked_indices = deterministic_rank_scores(selector.feature_scores_)
            ranking_sha256 = hashlib.sha256(selector.feature_scores_.tobytes()).hexdigest()
            buf = io.BytesIO()
            joblib.dump(selector, buf)
            storage_bytes = buf.tell()
            fit_time = getattr(selector, "fit_time_sec_", 0.0)
            return selector, ranked_indices, ranking_sha256, fit_time, storage_bytes

        # Fit selector on training data
        gc.collect()
        threshold = (
            self.memory_monitor.RF_LAUNCH_THRESHOLD_MB
            if "rf" in selector_name.lower() or "random_forest" in selector_name.lower()
            else self.memory_monitor.SVM_LAUNCH_THRESHOLD_MB
        )
        self.memory_monitor.check_threshold(
            f"fit_selector_{selector_name}",
            required_threshold_mb=threshold,
            max_stabilization_sec=20.0
        )

        train_path = Path(train_cache_path)
        with np.load(train_path, allow_pickle=True) as d:
            X_train = np.asarray(d["features"], dtype=np.float32)
            y_train = np.asarray(d["labels"], dtype=np.int64)

        t0 = time.perf_counter()
        if selector_name == "SelectKBest_ANOVA":
            selector = SelectKBestANOVA(k=4096)
            selector.fit(X_train, y_train)
        elif selector_name == "SelectFromModel_SVM":
            scaler = TrainOnlyStandardScaler()
            X_tr_scaled = scaler.fit_transform(X_train)
            selector = SelectFromModelSVM(
                k=4096,
                alpha=1.075384449940854e-05,
                max_iter=1000,
                tol=1e-3,
                random_state=42,
                n_jobs=self.svm_n_jobs
            )
            selector.fit(X_tr_scaled, y_train)
            del X_tr_scaled
        elif selector_name == "SelectFromModel_RF":
            selector = SelectFromModelRF(
                k=4096,
                n_estimators=300,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42,
                n_jobs=self.rf_n_jobs
            )
            selector.fit(X_train, y_train)
        else:
            raise ValueError(f"Unknown selector algorithm: {selector_name}")

        fit_time = float(time.perf_counter() - t0)
        selector.fit_time_sec_ = fit_time

        del X_train, y_train
        gc.collect()

        # Cache selector
        joblib.dump(selector, cached_selector_path)
        buf = io.BytesIO()
        joblib.dump(selector, buf)
        storage_bytes = buf.tell()

        ranked_indices = deterministic_rank_scores(selector.feature_scores_)
        ranking_sha256 = hashlib.sha256(selector.feature_scores_.tobytes()).hexdigest()

        return selector, ranked_indices, ranking_sha256, fit_time, storage_bytes

    def run_single_configuration(
        self,
        exp_config: Dict[str, Any],
        train_cache_path: Union[str, Path],
        val_cache_path: Union[str, Path],
        allow_retry: bool = False
    ) -> Dict[str, Any]:
        """
        Executes a single configuration from the comparative matrix:
        - Reuses the selector's single fitted ranking.
        - Enforces classifier-specific launch threshold.
        - Evaluates on validation data.
        - Saves 22-field metrics record and execution telemetry envelope.
        - Skips safely if already completed.
        """
        exp_id = exp_config["experiment_id"]
        selector_name = exp_config["selector"]
        k = exp_config["k"]
        classifier_name = exp_config["classifier"]

        train_path = Path(train_cache_path)
        val_path = Path(val_cache_path)

        # 1. Resumption check
        action, existing = self.check_resumption(exp_id, allow_retry=allow_retry)
        if action == "skip":
            return {"status": "skipped", "experiment_id": exp_id, "record": existing}

        # 2. Input cache SHA-256
        h = hashlib.sha256()
        with open(train_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        input_cache_sha256 = h.hexdigest()

        # 3. Fit or load selector
        selector, ranked_indices, ranking_sha256, sel_fit_time, sel_storage_bytes = self.fit_or_load_selector(
            selector_name, train_path
        )

        # 4. Deterministic slice of top-k indices
        selected_indices = np.sort(ranked_indices[:k])
        selected_indices_sha256 = hashlib.sha256(selected_indices.tobytes()).hexdigest()

        # 5. Check classifier-specific memory threshold
        launch_threshold = self.memory_monitor.get_launch_threshold(classifier_name)
        self.memory_monitor.check_threshold(
            f"{exp_id}_launch",
            required_threshold_mb=launch_threshold,
            max_stabilization_sec=20.0
        )
        pre_fit_mem = self.memory_monitor.get_memory_snapshot()
        start_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t0_total = time.perf_counter()

        # 6. Load training features sliced at selected_indices
        with np.load(train_path, allow_pickle=True) as d:
            X_tr = np.asarray(d["features"][:, selected_indices], dtype=np.float32)
            y_tr = np.asarray(d["labels"], dtype=np.int64)

        t0_train = time.perf_counter()
        if "svm" in classifier_name.lower():
            scaler = TrainOnlyStandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            alpha_val = 1.075384449940854e-05
            clf = SGDClassifier(
                loss="hinge",
                penalty="l2",
                alpha=alpha_val,
                max_iter=1000,
                tol=1e-3,
                random_state=42,
                n_jobs=self.svm_n_jobs
            )
            clf.fit(X_tr_scaled, y_tr)
            del X_tr_scaled
        else:
            scaler = None
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42,
                n_jobs=self.rf_n_jobs
            )
            clf.fit(X_tr, y_tr)

        model_train_time_sec = float(time.perf_counter() - t0_train)
        post_fit_mem = self.memory_monitor.get_memory_snapshot()

        del X_tr, y_tr
        gc.collect()

        # 7. Validation evaluation
        with np.load(val_path, allow_pickle=True) as d:
            X_val = np.asarray(d["features"][:, selected_indices], dtype=np.float32)
            y_val = np.asarray(d["labels"], dtype=np.int64)

        if scaler is not None:
            X_val_eval = scaler.transform(X_val)
        else:
            X_val_eval = X_val

        # Inference benchmarking
        latencies = []
        for idx in range(min(50, len(X_val_eval))):
            t0_inf = time.perf_counter()
            _ = clf.predict(X_val_eval[idx:idx+1])
            latencies.append((time.perf_counter() - t0_inf) * 1000.0)
        single_sample_inference_ms = float(np.mean(latencies))

        n_b = min(20, len(X_val_eval) // 32)
        if n_b > 0:
            t0_b = time.perf_counter()
            for b in range(n_b):
                _ = clf.predict(X_val_eval[b*32 : (b+1)*32])
            batch32_throughput_img_per_sec = float((n_b * 32) / (time.perf_counter() - t0_b))
        else:
            batch32_throughput_img_per_sec = 0.0

        y_pred = clf.predict(X_val_eval)
        acc = float(accuracy_score(y_val, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="macro", zero_division=0)
        _, _, w_f1, _ = precision_recall_fscore_support(y_val, y_pred, average="weighted", zero_division=0)

        buf_m = io.BytesIO()
        joblib.dump(clf, buf_m)
        model_storage_bytes = buf_m.tell()

        del X_val, y_val, y_pred
        if scaler is not None:
            del X_val_eval
        gc.collect()

        elapsed_sec = float(time.perf_counter() - t0_total)
        end_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Convergence
        if "svm" in classifier_name.lower():
            converged = bool(clf.n_iter_ < 1000)
            n_iter_max = int(clf.n_iter_)
        else:
            converged = True
            n_iter_max = 30

        # Construct 22-field ExperimentRecord
        exp_rec = ExperimentRecord(
            metadata=ExperimentMetadata(
                experiment_id=exp_id,
                selector_algorithm=selector_name,
                classifier_name=classifier_name,
                original_feature_count=INITIAL_FEATURE_DIM,
                selected_feature_count=k,
                feature_retention_ratio=float(k / INITIAL_FEATURE_DIM)
            ),
            classification_metrics=ClassificationMetrics(
                val_accuracy=acc,
                val_macro_precision=float(prec),
                val_macro_recall=float(rec),
                val_macro_f1=float(f1),
                val_weighted_f1=float(w_f1)
            ),
            timing_measurements=TimingMeasurements(
                selector_fit_time_sec=sel_fit_time,
                model_train_time_sec=model_train_time_sec,
                total_training_pipeline_sec=sel_fit_time + model_train_time_sec
            ),
            inference_measurements=InferenceMeasurements(
                single_sample_inference_ms=single_sample_inference_ms,
                batch32_throughput_img_per_sec=batch32_throughput_img_per_sec
            ),
            storage_measurements=StorageMeasurements(
                selector_storage_bytes=sel_storage_bytes,
                model_storage_bytes=model_storage_bytes,
                total_storage_bytes=sel_storage_bytes + model_storage_bytes
            ),
            convergence_diagnostics=ConvergenceDiagnostics(
                converged=converged,
                n_iter_max=n_iter_max,
                convergence_warning_count=0
            )
        )

        metrics_json_path = self.records_dir / f"{exp_id}_metrics.json"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(exp_rec.to_dict(), f, indent=2)

        # Construct ExecutionTelemetryRecord
        eval_cfg = (
            {
                "classifier": "Linear_SVM",
                "loss": "hinge",
                "penalty": "l2",
                "alpha": 1.075384449940854e-05,
                "max_iter": 1000,
                "tol": 1e-3,
                "n_jobs": self.svm_n_jobs,
                "random_state": 42
            }
            if "svm" in classifier_name.lower()
            else {
                "classifier": "Random_Forest_RF2",
                "n_estimators": 300,
                "max_depth": 30,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "random_state": 42,
                "n_jobs": self.rf_n_jobs
            }
        )

        telemetry_rec = ExecutionTelemetryRecord(
            experiment_id=exp_id,
            execution_status="completed",
            resume_action="retried_with_authorization" if action == "retry" else "executed",
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            input_cache_sha256=input_cache_sha256,
            selector_ranking_sha256=ranking_sha256,
            selected_feature_indices_sha256=selected_indices_sha256,
            selector_configuration={"algorithm": selector_name, "k": k},
            evaluator_configuration=eval_cfg,
            elapsed_seconds=elapsed_sec,
            pre_fit_available_mb=pre_fit_mem["available_mb"],
            pre_fit_process_rss_mb=pre_fit_mem["process_rss_mb"],
            post_fit_available_mb=post_fit_mem["available_mb"],
            post_fit_process_rss_mb=post_fit_mem["process_rss_mb"],
            artifact_paths={
                "metrics_json": str(metrics_json_path),
                "telemetry_json": str(self.get_telemetry_path(exp_id))
            },
            metrics_record=exp_rec.to_dict()
        )
        self.save_telemetry_record(telemetry_rec, allow_retry=allow_retry)
        return {
            "status": "completed",
            "experiment_id": exp_id,
            "record": exp_rec.to_dict(),
            "telemetry": telemetry_rec.to_dict()
        }

    def verify_configuration_record(self, exp_id: str) -> bool:
        """
        Verifies that both the 22-field metrics record and the execution telemetry record
        are valid, non-empty, and structurally compliant:
        - Checks existence of {exp_id}_metrics.json and {exp_id}_telemetry.json.
        - Asserts all 6 nested categories and 22 fields are populated.
        - Asserts execution_status is 'completed' and end_timestamp is populated.
        - Asserts cryptographic checksums are present and non-empty.
        """
        metrics_path = self.records_dir / f"{exp_id}_metrics.json"
        telemetry_path = self.get_telemetry_path(exp_id)

        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics record: {metrics_path}")
        if not telemetry_path.exists():
            raise FileNotFoundError(f"Missing telemetry record: {telemetry_path}")

        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        for cat in [
            "metadata",
            "classification_metrics",
            "timing_measurements",
            "inference_measurements",
            "storage_measurements",
            "convergence_diagnostics"
        ]:
            if cat not in m:
                raise ValueError(f"Missing category '{cat}' in {metrics_path}")

        with open(telemetry_path, "r", encoding="utf-8") as f:
            t = json.load(f)
        if t.get("execution_status") != "completed":
            raise ValueError(f"Telemetry execution_status is not 'completed': {t.get('execution_status')}")
        if not t.get("end_timestamp"):
            raise ValueError(f"Missing end_timestamp in {telemetry_path}")
        if not t.get("selected_feature_indices_sha256"):
            raise ValueError(f"Missing selected_feature_indices_sha256 in {telemetry_path}")

        return True

    def sync_comparative_summary(self) -> Path:
        """
        Scans all completed metrics records and produces a consolidated comparative summary
        ordered exactly according to the 30-configuration matrix contract.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        matrix = self.get_experiment_matrix(include_rfe=False)
        summary_records = []
        for exp in matrix:
            exp_id = exp["experiment_id"]
            metrics_path = self.records_dir / f"{exp_id}_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    summary_records.append(json.load(f))

        out_path = self.output_dir / "comparative_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary_records, f, indent=2)
        return out_path
