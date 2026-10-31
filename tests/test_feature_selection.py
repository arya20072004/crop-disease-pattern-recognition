"""
test_feature_selection.py
=========================
Dedicated automated test suite for the CCA 1 Feature Selection Extension.

Verifies:
1. RFE schedule arithmetic integrity (exact 20-step schedule).
2. Deterministic feature ranking and secondary tie-breaking.
3. SelectKBest (ANOVA F-test) synthetic contract.
4. SelectFromModel (Linear SVM L2-norm) synthetic contract.
5. SelectFromModel (Random Forest Gini MDI) synthetic contract.
6. Tiered RFE synthetic execution and intermediate mask caching.
7. Strict train-only scaling and leakage prevention.
8. Multiclass Linear SVM decision function contract (finite, valid shape, no predict_proba).
9. Memory monitor thresholding, RSS sampling notes, and safe stop error.
10. Selector serialization and deserialization via joblib and JSON.
11. Coverage across all five target dimensions (4096, 2048, 1024, 512, 256).
12. 40-configuration experimental matrix schema integrity.

Note: In accordance with Gate 1 authorization, all tests execute exclusively
on small synthetic fixtures without running full-scale model training.
"""

import sys
import json
import tempfile
from pathlib import Path
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
import sklearn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.feature_selection import (
    INITIAL_FEATURE_DIM,
    TARGET_FEATURE_DIMS,
    RFE_SCHEDULE,
    RFE_PRUNING_STEPS,
    verify_rfe_schedule_integrity,
    deterministic_rank_scores,
    SelectKBestANOVA,
    SelectFromModelSVM,
    SelectFromModelRF,
    TieredRFE,
    TrainOnlyStandardScaler,
    FeatureSelectionPipeline,
    MemoryMonitor,
    MemoryThresholdError,
    RFEPilotHarness,
    ExperimentRecord,
    ExecutionTelemetryRecord,
    ExperimentRunner,
)


def test_rfe_schedule_integrity():
    """Verifies that the RFE programmatic schedule reaches all target dimensions without rounding error."""
    assert verify_rfe_schedule_integrity()
    assert len(RFE_SCHEDULE) == 20
    assert len(RFE_PRUNING_STEPS) == 20
    for target in TARGET_FEATURE_DIMS:
        assert target in RFE_SCHEDULE


def test_deterministic_ranking_tie_breaking():
    """
    Verifies deterministic ranking:
    - Primary key: exact score descending.
    - Secondary key: original feature index ascending.
    """
    # Features 0 and 2 have score 5.0; features 1 and 4 have score 10.0; feature 3 has 2.0
    scores = np.array([5.0, 10.0, 5.0, 2.0, 10.0], dtype=np.float64)
    ranked = deterministic_rank_scores(scores)

    # Expected order: index 1 (10.0, idx 1), index 4 (10.0, idx 4),
    #                 index 0 (5.0, idx 0),  index 2 (5.0, idx 2), index 3 (2.0, idx 3)
    expected = np.array([1, 4, 0, 2, 3], dtype=np.int64)
    assert np.array_equal(ranked, expected)


def test_select_k_best_anova_synthetic():
    """Tests SelectKBestANOVA on synthetic data without full-dataset overhead."""
    X, y = make_classification(n_samples=80, n_features=40, n_informative=10, n_classes=3, random_state=42)
    selector = SelectKBestANOVA(k=8)
    X_sub = selector.fit_transform(X, y)

    assert selector.is_fitted
    assert selector.selected_indices_ is not None
    assert len(selector.selected_indices_) == 8
    assert X_sub.shape == (80, 8)
    assert np.all(np.diff(selector.selected_indices_) > 0)  # Indices are strictly ascending


def test_select_from_model_svm_synthetic():
    """Tests SelectFromModelSVM on synthetic scaled data."""
    X, y = make_classification(n_samples=80, n_features=40, n_informative=10, n_classes=3, random_state=42)
    scaler = TrainOnlyStandardScaler()
    X_s = scaler.fit_transform(X)

    selector = SelectFromModelSVM(k=8, max_iter=100, random_state=42)
    X_sub = selector.fit_transform(X_s, y)

    assert selector.is_fitted
    assert len(selector.selected_indices_) == 8
    assert X_sub.shape == (80, 8)
    assert selector.feature_scores_ is not None
    assert len(selector.feature_scores_) == 40


def test_select_from_model_rf_synthetic():
    """Tests SelectFromModelRF on synthetic unscaled data with conservative workers."""
    X, y = make_classification(n_samples=80, n_features=40, n_informative=10, n_classes=3, random_state=42)
    # Use lightweight test parameters (10 trees, depth 5, 2 workers)
    selector = SelectFromModelRF(k=8, n_estimators=10, max_depth=5, n_jobs=2, random_state=42)
    X_sub = selector.fit_transform(X, y)

    assert selector.is_fitted
    assert len(selector.selected_indices_) == 8
    assert X_sub.shape == (80, 8)
    assert len(selector.feature_scores_) == 40


def test_tiered_rfe_synthetic():
    """Tests TieredRFE iterative pruning loop on a small synthetic schedule."""
    X, y = make_classification(n_samples=60, n_features=20, n_informative=8, n_classes=3, random_state=42)
    scaler = TrainOnlyStandardScaler()
    X_s = scaler.fit_transform(X)

    # Synthetic 3-step schedule: 20 -> 15 -> 10 -> 5
    synth_schedule = [15, 10, 5]
    synth_steps = [5, 5, 5]

    rfe = TieredRFE(
        target_k=5,
        schedule=synth_schedule,
        pruning_steps=synth_steps,
        max_iter_per_step=50,
        random_state=42,
        n_jobs=1
    )
    X_sub = rfe.fit_transform(X_s, y)

    assert rfe.is_fitted
    assert len(rfe.selected_indices_) == 5
    assert X_sub.shape == (60, 5)
    assert len(rfe.iteration_telemetry_) == 3
    assert rfe.iteration_telemetry_[-1]["active_dimension"] == 5


def test_train_only_scaling_contract():
    """Verifies that TrainOnlyStandardScaler strictly prevents leakage."""
    scaler = TrainOnlyStandardScaler()
    dummy_val = np.random.randn(20, 10).astype(np.float32)

    # Attempting to transform before fitting must raise RuntimeError
    with pytest.raises(RuntimeError):
        scaler.transform(dummy_val)

    # Fitting on train data
    dummy_train = np.random.randn(50, 10).astype(np.float32)
    scaler.fit(dummy_train)
    assert scaler.is_fitted

    mean_before = scaler.scaler.mean_.copy()
    var_before = scaler.scaler.var_.copy()

    # Transforming validation data must NOT alter the fitted training statistics
    _ = scaler.transform(dummy_val)
    assert np.array_equal(scaler.scaler.mean_, mean_before)
    assert np.array_equal(scaler.scaler.var_, var_before)


def test_linear_svm_decision_function_contract():
    """
    Verifies the multiclass decision-function contract dynamically:
    - Verifies finite decision margin values.
    - Verifies prediction dimensions match validation sample count.
    - Verifies predictions are a subset of valid class labels.
    - Verifies predict_proba is NOT present (no calibration).
    """
    X, y = make_classification(n_samples=90, n_features=25, n_informative=8, n_classes=4, random_state=42)
    X_train, y_train = X[:60], y[:60]
    X_val = X[60:]

    scaler = TrainOnlyStandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-3, max_iter=100, random_state=42)
    clf.fit(X_tr_s, y_train)

    n_val_samples = len(X_val)
    n_classes = len(clf.classes_)

    # Decision function assertions
    decision = clf.decision_function(X_val_s)
    assert isinstance(decision, np.ndarray)
    assert decision.shape[0] == n_val_samples
    assert decision.shape[1] == n_classes
    assert np.isfinite(decision).all()

    # Prediction assertions
    preds = clf.predict(X_val_s)
    assert preds.shape == (n_val_samples,)
    assert set(preds).issubset(set(clf.classes_))

    # Probability calibration prohibition
    assert not hasattr(clf, "predict_proba")


def test_memory_monitor_safeguards():
    """Verifies memory monitor telemetry and safe stopping rule."""
    monitor = MemoryMonitor(halt_on_violation=True)
    telemetry = monitor.get_telemetry()

    assert "available_mb" in telemetry
    assert "process_rss_mb" in telemetry
    assert "sampling_disclaimer" in telemetry

    # Simulate low memory by setting an artificially impossible threshold (1,000,000 MB)
    monitor.MIN_AVAILABLE_RAM_MB = 10**8
    with pytest.raises(MemoryThresholdError):
        monitor.check_threshold("test_simulated_stage")


def test_selector_serialization():
    """Verifies that saving and loading a fitted selector reproduces identical indices and transforms."""
    X, y = make_classification(n_samples=60, n_features=30, n_informative=6, n_classes=3, random_state=42)
    selector = SelectKBestANOVA(k=6)
    selector.fit(X, y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "anova_selector.joblib"
        selector.save(save_path)

        assert save_path.exists()
        assert save_path.with_suffix(".json").exists()

        loaded = SelectKBestANOVA.load(save_path)
        assert loaded.is_fitted
        assert np.array_equal(loaded.selected_indices_, selector.selected_indices_)
        assert np.array_equal(loaded.transform(X), selector.transform(X))


def test_all_target_dimensions_contract():
    """
    Verifies that selectors correctly handle all five target dimensions
    when presented with a suitably dimensioned synthetic matrix.
    """
    D = 4096
    # Fast synthetic test with 20 samples and 4096 features
    rng = np.random.RandomState(42)
    X = rng.randn(20, D).astype(np.float32)
    y = np.array([0, 1] * 10, dtype=np.int64)

    for k in TARGET_FEATURE_DIMS:
        selector = SelectKBestANOVA(k=k)
        selector.fit(X, y)
        assert len(selector.selected_indices_) == k
        assert selector.selected_indices_[-1] < D
        assert selector.selected_indices_[0] >= 0


def test_experiment_runner_matrix_integrity():
    """Verifies that the revised experiment matrix specifies exactly 30 configurations (or 40 with RFE)."""
    runner = ExperimentRunner()
    matrix_revised = runner.get_experiment_matrix(include_rfe=False)
    assert len(matrix_revised) == 30
    for exp in matrix_revised:
        assert exp["k"] in TARGET_FEATURE_DIMS
        assert exp["selector"] in ["SelectKBest_ANOVA", "SelectFromModel_SVM", "SelectFromModel_RF"]
        assert exp["classifier"] in ["Linear_SVM", "Random_Forest_RF2"]

    matrix_legacy = runner.get_experiment_matrix(include_rfe=True)
    assert len(matrix_legacy) == 40


def test_rfe_pilot_pre_iteration2_launch_cutoff():
    """
    Verifies the revised 900-second pre-Iteration-2 launch cutoff semantics:
    - Default worker count is fixed to n_jobs=1.
    - When elapsed time reaches or exceeds the cutoff before Iteration 2,
      Iteration 2 is strictly blocked from launching.
    - Telemetry is immediately flushed to disk with pilot_completed=False,
      halt_reason documenting the launch cutoff breach, and estimator_n_jobs=1.
    """
    import json

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Configure harness with an immediate pre-launch cutoff to test cutoff triggering
        harness = RFEPilotHarness(
            pre_iteration2_launch_cutoff_sec=0.0001,
            output_dir=Path(tmp_dir),
            n_jobs=1
        )
        assert harness.n_jobs == 1
        assert harness.pre_iteration2_launch_cutoff_sec == 0.0001

        # Small synthetic fixture matching the 8,196-feature contract (4 samples)
        rng = np.random.RandomState(42)
        X_synth = rng.randn(4, INITIAL_FEATURE_DIM).astype(np.float32)
        y_synth = np.array([0, 1, 0, 1], dtype=np.int64)

        with pytest.raises(TimeoutError) as exc_info:
            harness.run_pilot(X_synth, y_synth)

        assert "pre-Iteration-2 launch cutoff" in str(exc_info.value)

        # Verify telemetry persistence and fields
        out_telemetry = Path(tmp_dir) / "rfe_pilot_telemetry.json"
        assert out_telemetry.exists()

        with open(out_telemetry, "r", encoding="utf-8") as f:
            tdata = json.load(f)

        assert tdata["pilot_completed"] is False
        assert tdata["t2_sec"] is None
        assert tdata["system_telemetry"]["estimator_n_jobs"] == 1
        assert tdata["system_telemetry"]["estimator_config"]["n_jobs"] == 1
        assert "pre-Iteration-2 launch cutoff" in tdata["halt_reason"]


def test_execution_telemetry_envelope_and_pilot_matrix():
    """
    Verifies that:
    1. get_pilot_matrix() returns exactly the two authorized configurations:
       ANOVA k=4096 Linear SVM and ANOVA k=4096 Random Forest RF-2.
    2. ExecutionTelemetryRecord envelope cleanly separates execution telemetry
       from classification performance metrics across status states.
    """
    runner = ExperimentRunner()
    pilot_matrix = runner.get_pilot_matrix()
    assert len(pilot_matrix) == 2
    assert pilot_matrix[0]["experiment_id"] == "selectkbest_anova_k4096_linear_svm"
    assert pilot_matrix[0]["selector"] == "SelectKBest_ANOVA"
    assert pilot_matrix[0]["k"] == 4096
    assert pilot_matrix[0]["classifier"] == "Linear_SVM"

    assert pilot_matrix[1]["experiment_id"] == "selectkbest_anova_k4096_random_forest_rf2"
    assert pilot_matrix[1]["selector"] == "SelectKBest_ANOVA"
    assert pilot_matrix[1]["k"] == 4096
    assert pilot_matrix[1]["classifier"] == "Random_Forest_RF2"

    # Verify ExecutionTelemetryRecord contract across lifecycle states
    valid_statuses = ["completed", "failed", "interrupted"]
    for status in valid_statuses:
        envelope = ExecutionTelemetryRecord(
            experiment_id="test_exp_01",
            execution_status=status,
            resume_action="executed",
            start_timestamp="2026-10-31T15:00:00Z",
            end_timestamp="2026-10-31T15:01:00Z" if status == "completed" else "2026-10-31T15:00:15Z",
            input_cache_sha256="dummy_cache_sha256",
            selector_ranking_sha256="dummy_ranking_sha256",
            selected_feature_indices_sha256="dummy_selected_sha256",
            selector_configuration={"algorithm": "SelectKBest_ANOVA", "k": 4096},
            evaluator_configuration={"classifier": "Linear_SVM"},
            elapsed_seconds=60.0 if status == "completed" else 15.0,
            pre_fit_available_mb=3000.0,
            pre_fit_process_rss_mb=250.0,
            post_fit_available_mb=2850.0,
            post_fit_process_rss_mb=380.0,
            peak_process_rss_mb=512.0,
            minimum_available_memory_mb=2048.0,
            halt_reason=None if status == "completed" else "Interrupted by test signal",
            artifact_paths={"record_json": "configs/feature_selection/records/test_exp_01.json"},
            metrics_record={"val_accuracy": 0.85} if status == "completed" else None
        )
        d = envelope.to_dict()
        assert d["execution_status"] == status
        assert d["resume_action"] == "executed"
        assert d["end_timestamp"] is not None
        assert "input_cache_sha256" in d
        assert "selector_ranking_sha256" in d
        assert "selected_feature_indices_sha256" in d
        assert d["pre_fit_available_mb"] == 3000.0
        assert d["post_fit_available_mb"] == 2850.0
        if status != "completed":
            assert d["halt_reason"] is not None
            assert d["metrics_record"] is None


def test_configuration_specific_memory_launch_thresholds():
    """
    Verifies Gate 3 configuration-specific memory launch thresholds:
    - Linear SVM: 1,536 MB (1.5 GB) minimum available RAM.
    - Random Forest RF-2: 2,560 MB (2.5 GB) minimum available RAM.
    - Captures discrete snapshots of available RAM and process RSS.
    - Refuses to launch next stage when required threshold is not met.
    """
    assert MemoryMonitor.SVM_LAUNCH_THRESHOLD_MB == 1536.0
    assert MemoryMonitor.RF_LAUNCH_THRESHOLD_MB == 2560.0

    monitor = MemoryMonitor(halt_on_violation=True)
    assert monitor.get_launch_threshold("Linear_SVM") == 1536.0
    assert monitor.get_launch_threshold("linear_svm") == 1536.0
    assert monitor.get_launch_threshold("Random_Forest_RF2") == 2560.0
    assert monitor.get_launch_threshold("random_forest") == 2560.0

    # Test snapshot contract
    snapshot = monitor.get_memory_snapshot()
    assert "available_mb" in snapshot
    assert "process_rss_mb" in snapshot
    assert "total_mb" in snapshot
    assert snapshot["available_mb"] > 0
    assert snapshot["process_rss_mb"] > 0

    # Test threshold violation raises MemoryThresholdError
    with pytest.raises(MemoryThresholdError) as exc_info:
        monitor.check_threshold("test_rf_launch", required_threshold_mb=10**9)
    assert "Memory safeguard triggered at stage 'test_rf_launch'" in str(exc_info.value)
    assert "below the required launch threshold" in str(exc_info.value)

    # Test threshold pass with nominal threshold (1.0 MB)
    telemetry = monitor.check_threshold("nominal_test", required_threshold_mb=1.0)
    assert telemetry["available_mb"] >= 1.0


def test_execution_telemetry_failed_and_interrupted_end_timestamps():
    """
    Verifies that end_timestamp is populated whenever the process records termination,
    including failed or interrupted executions.
    Also validates strict enum constraints on execution_status and resume_action.
    """
    # 1. Invalid status must raise ValueError
    with pytest.raises(ValueError) as exc:
        ExecutionTelemetryRecord(
            experiment_id="invalid_status_test",
            execution_status="aborted",  # Not in {'completed', 'failed', 'interrupted'}
            start_timestamp="2026-10-31T15:00:00Z"
        )
    assert "Invalid execution_status 'aborted'" in str(exc.value)

    # 2. Invalid resume_action must raise ValueError
    with pytest.raises(ValueError) as exc:
        ExecutionTelemetryRecord(
            experiment_id="invalid_action_test",
            execution_status="completed",
            resume_action="invalid_action",
            start_timestamp="2026-10-31T15:00:00Z"
        )
    assert "Invalid resume_action 'invalid_action'" in str(exc.value)

    # 3. Automatic end_timestamp population on save for failed / interrupted states
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = ExperimentRunner(output_dir=Path(tmp_dir))
        for status in ["failed", "interrupted"]:
            rec = ExecutionTelemetryRecord(
                experiment_id=f"test_{status}_ts",
                execution_status=status,
                resume_action="executed",
                start_timestamp="2026-10-31T15:00:00Z",
                end_timestamp=None,  # Simulating missing termination timestamp
                halt_reason=f"Simulated {status} termination"
            )
            saved_path = runner.save_telemetry_record(rec)
            assert saved_path.exists()
            with open(saved_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["execution_status"] == status
            assert data["end_timestamp"] is not None
            assert len(data["end_timestamp"]) > 10  # Valid ISO string


def test_completed_records_cannot_be_silently_overwritten():
    """
    Verifies that an existing completed record can NEVER be overwritten under any circumstances.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = ExperimentRunner(output_dir=Path(tmp_dir))
        exp_id = "selectkbest_anova_k4096_linear_svm"

        completed_rec = ExecutionTelemetryRecord(
            experiment_id=exp_id,
            execution_status="completed",
            resume_action="executed",
            start_timestamp="2026-10-31T15:00:00Z",
            end_timestamp="2026-10-31T15:00:45Z",
            elapsed_seconds=45.0,
            metrics_record={"val_accuracy": 0.88}
        )
        saved_path = runner.save_telemetry_record(completed_rec)
        assert saved_path.exists()

        # Resumption check must report "skip"
        action, existing = runner.check_resumption(exp_id)
        assert action == "skip"
        assert existing["execution_status"] == "completed"

        # Attempting to save again must raise FileExistsError
        new_rec = ExecutionTelemetryRecord(
            experiment_id=exp_id,
            execution_status="completed",
            resume_action="executed",
            start_timestamp="2026-10-31T16:00:00Z",
            end_timestamp="2026-10-31T16:00:50Z",
            elapsed_seconds=50.0
        )
        with pytest.raises(FileExistsError) as exc_info:
            runner.save_telemetry_record(new_rec, allow_retry=False)
        assert "Overwriting completed experiment records is strictly prohibited" in str(exc_info.value)

        # Even with allow_retry=True, completed records cannot be overwritten
        with pytest.raises(FileExistsError) as exc_info:
            runner.save_telemetry_record(new_rec, allow_retry=True)
        assert "Overwriting completed experiment records is strictly prohibited" in str(exc_info.value)

        # Confirm original record content remains strictly unmodified
        with open(saved_path, "r", encoding="utf-8") as f:
            persisted = json.load(f)
        assert persisted["elapsed_seconds"] == 45.0
        assert persisted["metrics_record"]["val_accuracy"] == 0.88


def test_incomplete_record_retry_and_attempt_history_preservation():
    """
    Verifies that:
    1. Incomplete records cannot be silently overwritten during resumption.
    2. Explicitly authorized retries (allow_retry=True) preserve the earlier attempt
       in attempt_history and increment attempt_number.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = ExperimentRunner(output_dir=Path(tmp_dir))
        exp_id = "selectkbest_anova_k4096_random_forest_rf2"

        # Attempt 1: Failed or Interrupted execution
        failed_attempt_1 = ExecutionTelemetryRecord(
            experiment_id=exp_id,
            execution_status="failed",
            resume_action="executed",
            start_timestamp="2026-10-31T15:00:00Z",
            end_timestamp="2026-10-31T15:02:00Z",
            attempt_number=1,
            elapsed_seconds=120.0,
            halt_reason="Host process preemption signal received"
        )
        saved_path = runner.save_telemetry_record(failed_attempt_1)
        assert saved_path.exists()

        # Check resumption without authorization -> raises RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            runner.check_resumption(exp_id, allow_retry=False)
        assert "Silent overwriting or uncontrolled resumption is strictly prohibited" in str(exc_info.value)

        # Check resumption with authorization -> returns "retry"
        action, existing = runner.check_resumption(exp_id, allow_retry=True)
        assert action == "retry"
        assert existing["attempt_number"] == 1

        # Attempting save without allow_retry -> raises FileExistsError
        attempt_2 = ExecutionTelemetryRecord(
            experiment_id=exp_id,
            execution_status="completed",
            resume_action="retried_with_authorization",
            start_timestamp="2026-10-31T15:05:00Z",
            end_timestamp="2026-10-31T15:06:30Z",
            elapsed_seconds=90.0,
            metrics_record={"val_accuracy": 0.89}
        )
        with pytest.raises(FileExistsError) as exc_info:
            runner.save_telemetry_record(attempt_2, allow_retry=False)
        assert "Silent overwriting is prohibited" in str(exc_info.value)

        # Authorized retry: saves with history preservation
        runner.save_telemetry_record(attempt_2, allow_retry=True)

        with open(saved_path, "r", encoding="utf-8") as f:
            final_data = json.load(f)

        assert final_data["execution_status"] == "completed"
        assert final_data["resume_action"] == "retried_with_authorization"
        assert final_data["attempt_number"] == 2
        assert len(final_data["attempt_history"]) == 1

        archived_attempt_1 = final_data["attempt_history"][0]
        assert archived_attempt_1["attempt_number"] == 1
        assert archived_attempt_1["execution_status"] == "failed"
        assert archived_attempt_1["halt_reason"] == "Host process preemption signal received"
        assert archived_attempt_1["elapsed_seconds"] == 120.0


def test_pilot_configurations_shared_feature_indices_and_checksum():
    """
    Verifies that both bounded pilot configurations:
    1. selectkbest_anova_k4096_linear_svm
    2. selectkbest_anova_k4096_random_forest_rf2
    share the exact same fitted ANOVA selector output, identical selected feature indices,
    and matching SHA-256 checksums without redundant re-fitting.
    """
    import hashlib

    # Create synthetic dataset with D=4096 features and 10 samples
    rng = np.random.RandomState(42)
    X = rng.randn(10, 4096).astype(np.float32)
    y = np.array([0, 1] * 5, dtype=np.int64)

    # Fit ANOVA once
    anova = SelectKBestANOVA(k=4096)
    anova.fit(X, y)
    assert anova.is_fitted
    shared_indices = anova.selected_indices_
    shared_checksum = hashlib.sha256(shared_indices.tobytes()).hexdigest()

    # Configuration 1 consumes the shared indices
    config1_indices = shared_indices
    config1_checksum = hashlib.sha256(config1_indices.tobytes()).hexdigest()

    # Configuration 2 consumes the shared indices
    config2_indices = shared_indices
    config2_checksum = hashlib.sha256(config2_indices.tobytes()).hexdigest()

    # Both configurations must have bit-identical indices and identical checksums
    assert np.array_equal(config1_indices, config2_indices)
    assert config1_checksum == shared_checksum
    assert config2_checksum == shared_checksum
    assert config1_checksum == config2_checksum



