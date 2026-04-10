# run_workflow.py
import sys

from matensemble.pipeline import Pipeline
from pathlib import Path

pipe = Pipeline()

script = Path(__file__).with_name("mpi_helloworld.py")

for i in range(1, 11):
    pipe.exec(command=[sys.executable, str(script), str(i)], num_tasks=50)

pipe.submit()
