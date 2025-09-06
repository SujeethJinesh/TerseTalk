from __future__ import annotations

import subprocess
import sys


if __name__ == "__main__":
  sys.exit(subprocess.call([sys.executable, "-m", "benchmarks.run_all", "--fast"]))

