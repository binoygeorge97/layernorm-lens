"""Load P-I's run.py under the unique module name "p1_run".

Other experiments have a run.py too (experiments/r6_exploratory/run.py), so a plain
`import run` can return the wrong module once another one is loaded (e.g. in the full
test suite). Every P-I script uses `p1_load.run()` instead.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run():
    """P-I's run.py as the module "p1_run" (loaded once, then reused)."""
    if "p1_run" not in sys.modules:
        spec = importlib.util.spec_from_file_location("p1_run", os.path.join(HERE, "run.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["p1_run"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["p1_run"]
