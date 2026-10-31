"""
feature_selection
=================
CCA 1 Pattern Recognition Feature Selection Extension Package.

Provides modular, leakage-free feature selection algorithms:
1. SelectKBest (ANOVA F-test / f_classif)
2. SelectFromModel (Linear SVM L2-norm)
3. SelectFromModel (Random Forest Gini MDI)
4. Tiered RFE (Linear SVM 20-step schedule)

All selectors adhere to strict train-only fitting contracts.
"""

from .selectors import (
    INITIAL_FEATURE_DIM,
    TARGET_FEATURE_DIMS,
    RFE_SCHEDULE,
    RFE_PRUNING_STEPS,
    verify_rfe_schedule_integrity,
    deterministic_rank_scores,
    BaseFeatureSelector,
    SelectKBestANOVA,
    SelectFromModelSVM,
    SelectFromModelRF,
    TieredRFE,
)

from .pipeline import (
    TrainOnlyStandardScaler,
    FeatureSelectionPipeline,
)

from .rfe_pilot import (
    RFEPilotHarness,
    RFEPilotTelemetry,
)

from .experiment_runner import (
    MemoryMonitor,
    MemoryThresholdError,
    ExperimentRecord,
    ExecutionTelemetryRecord,
    ExperimentRunner,
)

__all__ = [
    "INITIAL_FEATURE_DIM",
    "TARGET_FEATURE_DIMS",
    "RFE_SCHEDULE",
    "RFE_PRUNING_STEPS",
    "verify_rfe_schedule_integrity",
    "deterministic_rank_scores",
    "BaseFeatureSelector",
    "SelectKBestANOVA",
    "SelectFromModelSVM",
    "SelectFromModelRF",
    "TieredRFE",
    "TrainOnlyStandardScaler",
    "FeatureSelectionPipeline",
    "RFEPilotHarness",
    "RFEPilotTelemetry",
    "MemoryMonitor",
    "MemoryThresholdError",
    "ExperimentRecord",
    "ExecutionTelemetryRecord",
    "ExperimentRunner",
]

