Chat with Chat:

okay, well is there a way that this can be done programmatically? Say the user defines some functions, could we make some decorators that handle all of the args and parsing them? 

Then we build a task object filled with subtasks?

Something like this:

```python
class Task:
    def __init__(
        self,
        id: int | str,
        subtasks: list[SubTask],
        cores_per_task: int,
        gpus_per_task: int,
        dependencies: list[int | str] | None = None,
    ) -> None:
        """
        task objects n
        """

        self.id = id
        self.command = command
        self.subtasks = subtasks
        self.cores_per_task = cores_per_task
        self.gpus_per_task = gpus_per_task

class SubTask:
    def __init__(self, command: str) -> None:
        self.command = commmand
```


Or if there is a more generic design for tasks/jobs that would be more user friendly


Yes — that’s exactly a good design direction, and it’s a common way to make native Flux feel “Pythonic” without actually submitting Python callables.

You can absolutely build:

decorators to register user functions as pipeline steps

a Task / SubTask DAG model

a runner script that knows how to execute one registered step by name

automatic arg/result serialization (JSON/pickle)

That gives users a nice API like “define functions + dependencies,” while under the hood you still submit Flux Jobspec commands.

The key idea

Native Flux can’t run a Python function object directly, but it can run:

python flux_worker.py --task-id mytask --subtask-id B --inputs ...

So your framework can:

Capture Python functions with decorators

Build a DAG (Task, SubTask)

Serialize inputs/results to files

Submit one Flux job per subtask using a generic worker script

A user-friendly design (recommended)
1) User defines functions with decorators

Example of what the user writes:

@subtask(name="A")
def make_data():
    return {"x": 10}

@subtask(name="B", depends_on=["A"])
def process_b(a_output):
    return {"b": a_output["x"] * 2}

@subtask(name="C", depends_on=["A"])
def process_c(a_output):
    return {"c": a_output["x"] + 5}

@subtask(name="D", depends_on=["B", "C"])
def combine(b_output, c_output):
    return {"result": b_output["b"] + c_output["c"]}

This is much nicer than forcing users to write CLI parsing.

2) Your framework stores metadata in SubTask objects

Your SubTask should hold more than command — it should hold:

name

func (Python callable, used only by the local worker/registry)

depends_on

resources (cores/gpus)

serialization config (json/pickle)

maybe env, timeout, etc.

Something like:

from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass
class SubTask:
    name: str
    func: Callable[..., Any]
    depends_on: list[str] = field(default_factory=list)
    cores: int = 1
    gpus: int = 0
    serializer: str = "json"
3) Task is the DAG container (good idea)

Your Task class is a good start. I’d tweak it a bit:

@dataclass
class Task:
    id: str
    subtasks: list[SubTask]
    cores_per_task: int = 1
    gpus_per_task: int = 0
    dependencies: list[str] | None = None  # optional task-level deps
    workdir: str | None = None

But if subtasks can each have different resources, put resources on SubTask (more flexible).

4) Use a single generic worker script

Instead of generating separate scripts, use one worker:

python flux_worker.py --pipeline-spec pipeline.json --subtask B --workdir /path/run123

The worker:

loads the pipeline spec / registry

finds subtask B

loads outputs of dependencies (A, etc.)

calls the Python function

writes result to results/B.json

How results get passed around (automatically)

Your framework can standardize this:

each subtask writes to: workdir/results/<subtask_name>.json

worker loads dependencies from those files

injects them into the function call

For the DAG:

A writes results/A.json

B and C read results/A.json

D reads results/B.json, results/C.json

Users don’t handle file paths manually — your framework does.

Decorator pattern example (minimal)

Here’s a clean starting point:

# pipeline_api.py
from dataclasses import dataclass, field
from typing import Callable, Any

REGISTRY = {}

@dataclass
class SubTaskDef:
    name: str
    func: Callable[..., Any]
    depends_on: list[str] = field(default_factory=list)
    cores: int = 1
    gpus: int = 0

def subtask(name: str | None = None, depends_on: list[str] | None = None, cores=1, gpus=0):
    def decorator(func: Callable[..., Any]):
        task_name = name or func.__name__
        REGISTRY[task_name] = SubTaskDef(
            name=task_name,
            func=func,
            depends_on=depends_on or [],
            cores=cores,
            gpus=gpus,
        )
        return func
    return decorator

User writes:

# user_pipeline.py
from pipeline_api import subtask

@subtask(name="A")
def A():
    return {"x": 10}

@subtask(name="B", depends_on=["A"])
def B(A):
    return {"b": A["x"] * 2}

@subtask(name="C", depends_on=["A"])
def C(A):
    return {"c": A["x"] + 5}

@subtask(name="D", depends_on=["B", "C"])
def D(B, C):
    return {"result": B["b"] + C["c"]}

Note the nice trick:

function parameter names match dependency names (A, B, C)

worker can inject them automatically

Generic worker (how the magic works)

The worker can:

import user_pipeline (which populates REGISTRY)

get the requested subtask

load dependency JSON files

call function with kwargs

save output

Pseudo-code:

# flux_worker.py
import argparse, json, importlib
from pathlib import Path
from pipeline_api import REGISTRY

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--module", required=True)      # e.g. user_pipeline
    p.add_argument("--subtask", required=True)     # e.g. B
    p.add_argument("--workdir", required=True)
    args = p.parse_args()

    importlib.import_module(args.module)  # populates REGISTRY
    sub = REGISTRY[args.subtask]

    workdir = Path(args.workdir)
    result_dir = workdir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {}
    for dep in sub.depends_on:
        with open(result_dir / f"{dep}.json") as f:
            kwargs[dep] = json.load(f)

    result = sub.func(**kwargs)

    with open(result_dir / f"{sub.name}.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()

That’s the whole pattern.

Then Flux submission becomes simple

Your workflow manager builds jobspecs like:

A:

python flux_worker.py --module user_pipeline --subtask A --workdir /run1

B:

python flux_worker.py --module user_pipeline --subtask B --workdir /run1

C:

python flux_worker.py --module user_pipeline --subtask C --workdir /run1

D:

python flux_worker.py --module user_pipeline --subtask D --workdir /run1

The dependency ordering is handled by your scheduler/manager (as you said, assume that part is handled).

Your Task/SubTask design: suggested improvements

Your sketch is good, but I’d make these changes:

SubTask should not just hold command

If you want user-defined Python functions, SubTask should store:

function metadata (or function name/module)

dependencies

resource needs

output path

Task should not store command

In your sketch, Task has self.command = command but command isn’t passed in. Also, for a DAG task, the top-level Task usually has many commands (one per subtask).

A stronger generic design
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ResourceSpec:
    cores: int = 1
    gpus: int = 0
    nodes: int = 1

@dataclass
class SubTask:
    id: str
    module: str                  # "user_pipeline"
    func_name: str               # "B" or actual function name
    depends_on: list[str] = field(default_factory=list)
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    serializer: str = "json"

@dataclass
class Task:
    id: str
    subtasks: dict[str, SubTask]   # easier lookup than list
    workdir: str

This is nice because SubTask is now serializable and schedulable.

Even more user-friendly: a Pipeline object

Instead of making users build Task objects manually, let decorators build a Pipeline:

pipeline = Pipeline("my_run")

@pipeline.subtask()
def A(): ...

@pipeline.subtask(depends_on=["A"])
def B(A): ...

Then:

pipeline.submit_with_flux()

This is usually the best UX.

One thing to watch out for

If users define functions in arbitrary files, the worker must be able to import that module on compute nodes. So you’ll want one of:

package installed in the environment

submit from a shared filesystem and set cwd

ship code with the job (less common)

In HPC/shared FS setups, this is usually fine.

Bottom line

Yes — your decorator + Task/SubTask idea is a very good approach.

The best pattern is:

decorators register functions + metadata

generic worker script executes one subtask by name

JSON files pass results

Flux Jobspec commands run the worker

scheduler enforces dependency order

It gives users a smooth Python API while staying compatible with native Flux.

If you want, I can draft a small working prototype of:

pipeline_api.py (decorators + registry)

flux_worker.py

user_pipeline.py

submit_dag.py (builds Jobspecs)




```python
def job(
    method: Callable | None = None, **job_kwargs
) -> Callable[..., Job] | Callable[..., Callable[..., Job]]:
    """
    Wrap a function to produce a :obj:`Job`.

    :obj:`Job` objects are delayed function calls that can be used in an
    :obj:`Flow`. A job is composed of the function name and source and any
    arguments for the function. This decorator makes it simple to create
    job objects directly from a function definition. See the examples for more details.

    Parameters
    ----------
    method
        A function to wrap. This should not be specified directly and is implied
        by the decorator.
    **job_kwargs
        Other keyword arguments that will get passed to the :obj:`Job` init method.

    Examples
    --------
    >>> @job
    ... def print_message():
    ...     print("I am a Job")
    >>> print_job = print_message()
    >>> type(print_job)
    <class 'jobflow.core.job.Job'>
    >>> print_job.function
    <function print_message at 0x7ff72bdf6af0>

    Jobs can have required and optional parameters.

    >>> @job
    ... def print_sum(a, b=0):
    ...     return print(a + b)
    >>> print_sum_job = print_sum(1, 2)
    >>> print_sum_job.function_args
    (1, )
    >>> print_sum_job.function_kwargs
    {"b": 2}

    If the function returns a value it can be referenced using the ``output``
    attribute of the job.

    >>> @job
    ... def add(a, b):
    ...     return a + b
    >>> add_task = add(1, 2)
    >>> add_task.output
    OutputReference('abeb6f48-9b34-4698-ab69-e4dc2127ebe9')

    .. Note::
        Because the task has not yet been run, the output value is an
        :obj:`OutputReference` object. References are automatically converted to their
        computed values (resolved) when the task runs.

    If a dictionary of values is returned, the values can be indexed in the usual
    way.

    >>> @job
    ... def compute(a, b):
    ...     return {"sum": a + b, "product": a * b}
    >>> compute_task = compute(1, 2)
    >>> compute_task.output["sum"]
    OutputReference('abeb6f48-9b34-4698-ab69-e4dc2127ebe9', 'sum')

    .. Warning::
        If an output is indexed incorrectly, for example by trying to access a key that
        doesn't exist, this error will only be raised when the Job is executed.

    Jobs can return :obj:`.Response` objects that control the flow execution flow.
    For example, to replace the current job with another job, ``replace`` can be used.

    >>> from jobflow import Response
    >>> @job
    ... def replace(a, b):
    ...     new_job = compute(a, b)
    ...     return Response(replace=new_job)

    By default, job outputs are stored in the :obj`.JobStore` ``docs_store``. However,
    the :obj:`.JobStore` `additional_stores`` can also be used for job outputs. The
    stores are specified as keyword arguments, where the argument name gives the store
    name and the argument value is the type of data/key to store in that store. More
    details on the accepted key types are given in the :obj:`Job` docstring. In the
    example below, the "graph" key is stored in an additional store named "graphs" and
    the "data" key is stored in an additional store called "large_data".

    >>> @job(large_data="data", graphs="graph")
    ... def compute(a, b):
    ...     return {"data": b, "graph": a }

    .. Note::
        Using additional stores requires the :obj:`.JobStore` to be configured with
        the required store names present. See the :obj:`.JobStore` docstring for more
        details.

    See Also
    --------
    Job, .Flow, .Response
    """

    def decorator(func):
        from functools import wraps

        # unwrap staticmethod or classmethod decorators
        desc = next(
            (desc for desc in (staticmethod, classmethod) if isinstance(func, desc)),
            None,
        )

        if desc:
            func = func.__func__

        @wraps(func)
        def get_job(*args, **kwargs) -> Job:
            f = func
            if len(args) > 0:
                # see if the first argument has a function with the same name as
                # this function
                met = getattr(args[0], func.__name__, None)
                if met:
                    # if so, check to see if that function ha been wrapped and
                    # whether the unwrapped function is the same as this function
                    wrap = getattr(met, "__func__", None)
                    if getattr(wrap, "original", None) is func:
                        # Ah ha. The function is a bound method.
                        f = met
                        args = args[1:]

            return Job(
                function=f, function_args=args, function_kwargs=kwargs, **job_kwargs
            )

        get_job.original = func

        if desc:
            # rewrap staticmethod or classmethod decorators
            get_job = desc(get_job)

        return get_job

    # See if we're being called as @job or @job().
    if method is None:
        # We're called with parens.
        return decorator

    # We're called as @job without parens.
    return decorator(method)
```
