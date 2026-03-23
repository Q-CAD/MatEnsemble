Tutorial
========

There are a couple of example workflows that you can reference on `GitHub <https://github.com/FredDude2004/Matensemble/tree/main/example_workflows>`_

Using MatEnsemble
-----------------

After you have installed MatEnsemble or built a container image that is ready to 
run using MatEnsemble is quite simple. The first thing to do is to define your 
:obj:`Jobs <Job>`. Which is done with the :obj:`Pipeline` There are two different Job
flavors. EXECUTABLE Jobs and PYTHON Jobs. We'll first go over the simpler of the
two. EXECUTABLE Jobs. 

.. code-block:: python

    from matensemble.pipeline import Pipeline

    pipe = Pipeline()

    pipe.exec(command=["/bin/echo", "hello from MatEnsemble"])

    pipe.submit()

The :obj:`Pipeline` is used to build your workflow. You can add as many jobs as 
you want to the pipeline. Here is an example of creating 10 of the same
:obj:`Jobs <Job>` 

.. code-block:: python

    import sys
    from pathlib import Path
    from matensemble.pipeline import Pipeline

    pipe = Pipeline()

    job_command = Path(__file__).with_name("mpi_helloworld.py")

    for _ in range(10):
        pipe.exec(
            name="hello",
            command=job_command,
            num_tasks=50,
            mpi=True,
        )

    pipe.submit()

Calling :meth:`pipe.exec()` only creates the :obj:`Job` object that the 
:obj:`Pipeline` uses to populate the job list. When you call :meth:`pipe.submit()`
that is when the DAG is created and the workflow will run. 


Python Jobs
-----------
The other flavor of Job that MatEnsemble offers are Python jobs. Python jobs 
are delayed function calls that get passed to Flux. To define a Python Job 
you can use the :obj:`Pipeline` to decorate a regular old python funciton. 

.. code-block:: python

    # workflow.py
    from mpi4py import MPI
    from matensemble.pipeline import Pipeline

    pipe = Pipeline()

    @pipe.job()
    def run_mpi_hello(task_id: int):
        size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
        name = MPI.Get_processor_name()

        with open(f"task_{task_id}_rank_{rank}.txt", "w") as f:
            f.write(f"Hello from rank {rank}/{size} on {name}, task={task_id}\n")

        return rank

This decorated python function when called will create a :obj:`Job` in the 
:obj:`Pipeline` and it will return an :obj:`OutputReference`. In order to run 
the workflow we can call :meth:`pipeline.submit()` as before with a little caveat. 


.. code-block:: python

    # run_workflow.py
    from workflow import pipe, run_mpi_hello

    def main():
        for i in range(1, 11):
            run_mpi_hello(i)

        pipe.submit()

    if __name__ == "__main__":
        main()

In order for MatEnsemble to work the decorated function needs to be in an importable
module that is **NOT** run as *__main__*. In other words you cannot run the module
where :meth:`pipeline.job()` is defined with:

.. code-block:: bash

   python <path_to_script.py>

This will cause the name of the module in the :obj:`Job` to be "__main__" rather
than the actual name of the module where you define your :obj:`Jobs <Job>`

.. warning::
   Do not define you jobs in a module that you run as "__main__" instead define
   your jobs in a seperate python module and have a runner script that you import
   the decorated jobs function and build the DAG. This is the current behavior of
   MatEnsemble but this may change in the future to be more user friendly. 

So for now it is required that you follow a structure similar to this for your 
workflow. 

.. code-block:: 

   .
   ├── functions.py
   └── run_workflow.py


Where you would create the :obj:`Pipeline`, define your functions and decorate
them with :meth:`pipeline.job()` in the *functions.py* module. In the *run_workflow.py*
module you would then import the pipline object that you created and all of the 
functions that you defined, and you would call them and build the workflow graph.
You would also be able to define your EXECUTABLE :obj:`Jobs <Job>` here as well.
Finally you would call pipeline.submit() in *run_workflow.py* and you would
run the script as "__main__". 
    


    

