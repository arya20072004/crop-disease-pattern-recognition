"""
run_evaluation_suite.py
=======================
Executes verification tests for the Crop Disease Pattern Recognition repository:
1. Validates dataset manifest structures and sample counts.
2. Validates classification metric schemas and numerical ranges.
3. Validates paired inference benchmark arithmetic monotonicity.
4. Verifies Random Forest feature importance distributions.
"""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():
    print("=" * 60)
    print("RUNNING REPOSITORY VERIFICATION SUITE")
    print("=" * 60)
    test_dir = REPO_ROOT / "tests"
    exit_code = pytest.main(["-v", str(test_dir)])
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
