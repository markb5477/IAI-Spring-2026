"""Discover and run every test_*.py in this folder.

Usage:
    python run_tests.py                    # from inside postulate_testing/
    python -m postulate_testing.run_tests  # not supported (parent isn't a package)
"""

from __future__ import annotations

import os
import sys
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    sys.path.insert(0, os.path.dirname(HERE))   # for: from formula import ...
    sys.path.insert(0, HERE)                    # for: import test_revision_...

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=HERE, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
