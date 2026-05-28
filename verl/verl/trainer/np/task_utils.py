"""NP reuses the ES task components verbatim (incl. the opd_math branch).

Importing from the ES module keeps a single source of truth for prompt
processors and reward functions. See verl/verl/trainer/es/task_utils.py.
"""
from verl.trainer.es.task_utils import get_task_components  # noqa: F401
