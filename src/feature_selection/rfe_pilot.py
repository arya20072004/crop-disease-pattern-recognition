"""
rfe_pilot.py
============
Two-Iteration Pilot Benchmark Harness for Recursive Feature Elimination (RFE).

Measures empirical telemetry over the first two pruning steps (8,196 -> 7,171 -> 6,146):
- Records exact wall-clock times t1 and t2.
- Records process RSS and system available memory telemetry.
- Evaluates the dimensionality-based runtime projection against the 900-second ceiling.
- Computes the illustrative unreduced runtime estimate.

Execution Governance:
- This module DOES NOT execute upon import.
- Execution requires explicit invocation of `run_pilot()`.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.exceptions import ConvergenceWarning

from .selectors import deterministic_rank_scores


def get_system_memory_info() -> Dict[str, float]:
    """
    Robust memory query returning available RAM, total RAM, and process RSS in MB.
    Uses psutil if installed; otherwise uses standard library ctypes on Windows.
    """
    try:
        import psutil
        vm = psutil.virtual_memory()
        rss = psutil.Process(os.getpid()).memory_info().rss
        return {
            "available_mb": round(vm.available / (1024 * 1024), 2),
            "total_mb": round(vm.total / (1024 * 1024), 2),
            "process_rss_mb": round(rss / (1024 * 1024), 2)
        }
    except ImportError:
        pass

    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.wintypes.DWORD),
                ("dwMemoryLoad", ctypes.wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        f = ctypes.windll.kernel32.K32GetProcessMemoryInfo
        f.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.wintypes.DWORD]
        f.restype = ctypes.wintypes.BOOL
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        f(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)

        return {
            "available_mb": round(stat.ullAvailPhys / (1024 * 1024), 2),
            "total_mb": round(stat.ullTotalPhys / (1024 * 1024), 2),
            "process_rss_mb": round(pmc.WorkingSetSize / (1024 * 1024), 2)
        }

    # Fallback default
    return {"available_mb": 4096.0, "total_mb": 16384.0, "process_rss_mb": 100.0}


@dataclass
class PilotStepTelemetry:
    iteration: int
    initial_dimension: int
    pruned_dimension: int
    active_dimension_after_step: int
    step_duration_sec: float
    converged: bool
    n_iter_max: int
    convergence_warning_count: int
    pre_fit_rss_mb: float
    post_fit_rss_mb: float
    available_memory_mb: float


@dataclass
class RFEPilotTelemetry:
    pilot_completed: bool
    t1_sec: float
    t2_sec: Optional[float]
    total_pilot_time_sec: float
    estimated_seconds_per_feature: Optional[float]
    projected_full_rfe_sec: Optional[float]
    illustrative_unreduced_rfe_sec: Optional[float]
    within_900s_limit: bool
    recommendation: str
    steps: List[Dict[str, Any]]
    system_telemetry: Dict[str, Any]
    methodological_notes: str
    halt_reason: Optional[str] = None


class RFEPilotHarness:
    """
    Executes and reports on the 2-iteration RFE pilot benchmark.
    Guarantees strict memory checks, 900-second pre-Iteration-2 launch cutoff, and flushes telemetry upon early halt.
    """

    MIN_AVAILABLE_RAM_MB: float = 1536.0  # 1.5 GB

    def __init__(
        self,
        max_iter: int = 250,
        tol: float = 1e-3,
        random_state: int = 42,
        n_jobs: int = 1,
        pre_iteration2_launch_cutoff_sec: float = 900.0,
        max_total_seconds: Optional[float] = None,
        output_dir: Optional[Path] = None
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_jobs = n_jobs
        # Support pre_iteration2_launch_cutoff_sec with backward compatibility for max_total_seconds argument
        self.pre_iteration2_launch_cutoff_sec = (
            max_total_seconds if max_total_seconds is not None else pre_iteration2_launch_cutoff_sec
        )
        self.output_dir = output_dir or (Path(__file__).resolve().parents[2] / "configs" / "feature_selection")

    def check_memory_safeguard(self) -> None:
        """Verifies that system available memory exceeds the 1.5 GB threshold."""
        mem = get_system_memory_info()
        if mem["available_mb"] < self.MIN_AVAILABLE_RAM_MB:
            raise RuntimeError(
                f"Memory threshold violation: available memory ({mem['available_mb']:.1f} MB) "
                f"is below the mandatory 1.5 GB threshold ({self.MIN_AVAILABLE_RAM_MB:.1f} MB). "
                f"Halting before RFE pilot execution."
            )

    def run_pilot(
        self,
        X_train_scaled: np.ndarray,
        y_train: np.ndarray
    ) -> RFEPilotTelemetry:
        """
        Executes exactly 2 iterations of RFE pruning (pruning 1,025 features each step).
        Requires standardized training features.
        """
        self.check_memory_safeguard()

        n_samples, initial_dim = X_train_scaled.shape
        assert initial_dim == 8196, f"Expected 8,196 features, got {initial_dim}"

        steps_telemetry: List[PilotStepTelemetry] = []
        active_indices = np.arange(initial_dim, dtype=np.int64)

        pilot_t0 = time.perf_counter()
        timings: List[float] = []

        # Pilot executes Step 1 (8,196 -> 7,171) and Step 2 (7,171 -> 6,146)
        pruning_schedule = [(7171, 1025), (6146, 1025)]

        for iter_num, (target_dim, prune_count) in enumerate(pruning_schedule, start=1):
            # 900-second pre-Iteration-2 launch cutoff check
            if iter_num == 2:
                elapsed_so_far = time.perf_counter() - pilot_t0
                if elapsed_so_far >= self.pre_iteration2_launch_cutoff_sec:
                    final_mem = get_system_memory_info()
                    telemetry_obj = RFEPilotTelemetry(
                        pilot_completed=False,
                        t1_sec=round(timings[0], 3) if timings else 0.0,
                        t2_sec=None,
                        total_pilot_time_sec=round(elapsed_so_far, 3),
                        estimated_seconds_per_feature=None,
                        projected_full_rfe_sec=None,
                        illustrative_unreduced_rfe_sec=round(20.0 * timings[0], 2) if timings else None,
                        within_900s_limit=False,
                        recommendation="ABORT_OR_REVISE_RFE",
                        steps=[asdict(s) for s in steps_telemetry],
                        system_telemetry={
                            "cpu_cores_logical": os.cpu_count(),
                            "estimator_n_jobs": self.n_jobs,
                            "estimator_config": {
                                "loss": "hinge",
                                "penalty": "l2",
                                "alpha": 1.075384e-05,
                                "max_iter": self.max_iter,
                                "tol": self.tol,
                                "random_state": self.random_state,
                                "n_jobs": self.n_jobs
                            },
                            "total_ram_mb": final_mem["total_mb"],
                            "available_ram_mb": final_mem["available_mb"],
                            "rss_sampling_method": "process_working_set_pre_post_fit",
                            "rss_sampling_limitation": "Sampled at discrete iteration boundaries; intra-iteration transient spikes are not captured."
                        },
                        methodological_notes=(
                            "Halted prior to starting Iteration 2 due to 900-second pre-Iteration-2 launch cutoff breach. "
                            "No automatic retry or continuation is permitted."
                        ),
                        halt_reason=(
                            f"Launch cutoff breach: elapsed time ({elapsed_so_far:.2f}s) reached or exceeded "
                            f"{self.pre_iteration2_launch_cutoff_sec}s pre-Iteration-2 launch cutoff."
                        )
                    )
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    with open(self.output_dir / "rfe_pilot_telemetry.json", "w", encoding="utf-8") as f:
                        json.dump(asdict(telemetry_obj), f, indent=2)
                    raise TimeoutError(telemetry_obj.halt_reason)

            # Pre-iteration memory safeguard check
            mem_check = get_system_memory_info()
            if mem_check["available_mb"] < self.MIN_AVAILABLE_RAM_MB:
                elapsed_so_far = time.perf_counter() - pilot_t0
                telemetry_obj = RFEPilotTelemetry(
                    pilot_completed=False,
                    t1_sec=round(timings[0], 3) if timings else 0.0,
                    t2_sec=None,
                    total_pilot_time_sec=round(elapsed_so_far, 3),
                    estimated_seconds_per_feature=None,
                    projected_full_rfe_sec=None,
                    illustrative_unreduced_rfe_sec=round(20.0 * timings[0], 2) if timings else None,
                    within_900s_limit=True,
                    recommendation="ABORT_OR_REVISE_RFE",
                    steps=[asdict(s) for s in steps_telemetry],
                    system_telemetry={
                        "cpu_cores_logical": os.cpu_count(),
                        "estimator_n_jobs": self.n_jobs,
                        "estimator_config": {
                            "loss": "hinge",
                            "penalty": "l2",
                            "alpha": 1.075384e-05,
                            "max_iter": self.max_iter,
                            "tol": self.tol,
                            "random_state": self.random_state,
                            "n_jobs": self.n_jobs
                        },
                        "total_ram_mb": mem_check["total_mb"],
                        "available_ram_mb": mem_check["available_mb"],
                        "rss_sampling_method": "process_working_set_pre_post_fit",
                        "rss_sampling_limitation": "Sampled at discrete iteration boundaries; intra-iteration transient spikes are not captured."
                    },
                    methodological_notes="Halted prior to starting iteration due to available RAM falling below 1.5 GB.",
                    halt_reason=(
                        f"Memory threshold violation: available memory ({mem_check['available_mb']:.1f} MB) "
                        f"is below mandatory threshold ({self.MIN_AVAILABLE_RAM_MB:.1f} MB)."
                    )
                )
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with open(self.output_dir / "rfe_pilot_telemetry.json", "w", encoding="utf-8") as f:
                    json.dump(asdict(telemetry_obj), f, indent=2)
                raise RuntimeError(telemetry_obj.halt_reason)

            pre_mem = mem_check

            X_curr = X_train_scaled[:, active_indices]
            start_dim = len(active_indices)

            eff_alpha = 1.0 / (float(n_samples) * 10.0)
            clf = SGDClassifier(
                loss="hinge",
                penalty="l2",
                alpha=eff_alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )

            step_t0 = time.perf_counter()
            converged = True
            warning_count = 0

            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                clf.fit(X_curr, y_train)
                for w in caught_warnings:
                    if issubclass(w.category, ConvergenceWarning):
                        converged = False
                        warning_count += 1

            step_duration = time.perf_counter() - step_t0
            timings.append(step_duration)

            post_mem = get_system_memory_info()

            # Compute L2 norm across classes
            l2_norms = np.sqrt(np.sum(clf.coef_ ** 2, axis=0)).astype(np.float64)
            local_ranked = deterministic_rank_scores(l2_norms)

            # Retain top (start_dim - prune_count)
            n_retain = start_dim - prune_count
            surviving_local = np.sort(local_ranked[:n_retain])
            active_indices = active_indices[surviving_local]

            steps_telemetry.append(PilotStepTelemetry(
                iteration=iter_num,
                initial_dimension=start_dim,
                pruned_dimension=prune_count,
                active_dimension_after_step=len(active_indices),
                step_duration_sec=round(step_duration, 3),
                converged=converged,
                n_iter_max=int(np.max(clf.n_iter_)) if hasattr(clf, "n_iter_") else 0,
                convergence_warning_count=warning_count,
                pre_fit_rss_mb=pre_mem["process_rss_mb"],
                post_fit_rss_mb=post_mem["process_rss_mb"],
                available_memory_mb=post_mem["available_mb"]
            ))

        total_pilot_duration = time.perf_counter() - pilot_t0
        t1, t2 = timings[0], timings[1]

        # Linear dimensionality-based projection (empirical approximation, not a runtime guarantee):
        # beta = 0.5 * (t1 / 8196 + t2 / 7171)
        # T_projected = t1 + t2 + beta * sum(D_i), for i = 3 through 20.
        # Dimensions trained on in iterations 3 through 20:
        remaining_dims = [
            6146, 5121, 4096, 3584, 3072, 2560, 2048, 1792, 1536, 1280,
            1024, 896, 768, 640, 512, 448, 384, 320
        ]
        assert len(remaining_dims) == 18, f"Expected 18 remaining iterations, got {len(remaining_dims)}"
        remaining_dims_sum = sum(remaining_dims)  # 36,227 features
        beta_hat = 0.5 * ((t1 / 8196.0) + (t2 / 7171.0))
        projected_remaining_time = beta_hat * remaining_dims_sum
        projected_total = t1 + t2 + projected_remaining_time

        # Illustrative unreduced estimate (illustrative estimate, not a proven upper bound):
        # T_illustrative = 20 * max(t1, t2)
        illustrative_unreduced = 20.0 * max(t1, t2)

        within_limit = projected_total <= 900.0
        recommendation = (
            "PROCEED_TO_FULL_RFE" if within_limit and all(s.converged for s in steps_telemetry)
            else "ABORT_OR_REVISE_RFE"
        )

        final_mem = get_system_memory_info()
        telemetry_obj = RFEPilotTelemetry(
            pilot_completed=True,
            t1_sec=round(t1, 3),
            t2_sec=round(t2, 3),
            total_pilot_time_sec=round(total_pilot_duration, 3),
            estimated_seconds_per_feature=float(f"{beta_hat:.6e}"),
            projected_full_rfe_sec=round(projected_total, 2),
            illustrative_unreduced_rfe_sec=round(illustrative_unreduced, 2),
            within_900s_limit=within_limit,
            recommendation=recommendation,
            steps=[asdict(s) for s in steps_telemetry],
            system_telemetry={
                "cpu_cores_logical": os.cpu_count(),
                "estimator_n_jobs": self.n_jobs,
                "estimator_config": {
                    "loss": "hinge",
                    "penalty": "l2",
                    "alpha": 1.075384e-05,
                    "max_iter": self.max_iter,
                    "tol": self.tol,
                    "random_state": self.random_state,
                    "n_jobs": self.n_jobs
                },
                "total_ram_mb": final_mem["total_mb"],
                "available_ram_mb": final_mem["available_mb"],
                "rss_sampling_method": "process_working_set_pre_post_fit",
                "rss_sampling_limitation": "Sampled at discrete iteration boundaries; intra-iteration transient spikes are not captured."
            },
            methodological_notes=(
                "Projected runtime is an empirical approximation assuming linear scaling of SGD computation with "
                "feature dimension D (beta = 0.5 * (t1 / 8196 + t2 / 7171); T_projected = t1 + t2 + beta * sum(D_i)). "
                "It is not a runtime guarantee. The illustrative unreduced estimate (20 * max(t1, t2)) is an "
                "illustrative estimate of constant-dimension fitting and is not a proven upper bound. "
                "Pre-Iteration-2 launch cutoff: The pilot enforces a 900-second launch cutoff before launching Iteration 2. "
                "An iteration already executing cannot necessarily be interrupted at the 900-second boundary; total pilot "
                "execution may exceed 900 seconds if Iteration 2 starts before the cutoff and finishes after. "
                "No automatic retry or continuation is permitted. "
                "Neither estimate guarantees runtime against OS scheduling jitter, system background load, or cache effects."
            )
        )

        # Serialize telemetry to disk
        self.output_dir.mkdir(parents=True, exist_ok=True)
        telemetry_path = self.output_dir / "rfe_pilot_telemetry.json"
        with open(telemetry_path, "w", encoding="utf-8") as f:
            json.dump(asdict(telemetry_obj), f, indent=2)

        return telemetry_obj
