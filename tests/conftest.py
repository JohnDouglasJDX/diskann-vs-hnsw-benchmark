"""Make the harness modules in ``scripts/`` importable from the tests."""
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
)
