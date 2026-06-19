import sys
import pytest

sys.path.insert(0, ".")
sys.exit(pytest.main(["-v", "tests/test_rewardio.py"]))
