Tutorial
========

There are a couple of example workflows that you can reference on `GitHub <https://github.com/FredDude2004/Matensemble/tree/main/example_workflows>`_

Using MatEnsemble
-----------------

After you have installed MatEnsemble or built a container image that is ready to 
run using MatEnsemble is quite simple. The first thing to do is to define your 
:obj:`Job`s. Which is done with the :obj:`Pipeline` There are two different Job
flavors. EXECUTABLE Jobs and PYTHON Jobs. We'll first go over the simpler of the
two. EXECUTABLE Jobs. 

.. code-block:: python

    import sys

    from matensemble.pipeline import Pipeline
    from pathlib import Path

    pipe = Pipeline()

    script = Path(__file__).with_name("mpi_helloworld.py")

    for i in range(1, 11):
        pipe.exec(command=[sys.executable, str(script), str(i)], num_tasks=50)

    pipe.submit()

The :obj:`Pipeline` is used to build your workflow. 

