==============
Overview Hello
==============

MatEnsemble is a workflow manager for running lots of similar simulation :obj:`Job`s 
on a supercomputer as efficiently as possible. MatEnsemble launches :obj:`Job`s,
watches them finish, records what happened, and immediately sends more when 
resources open up. 

High Throughput Computing
-------------------------
High Throughput Computing (HTC) is a computing paradigm designed to maximize 
the total amount of computational work completed over long periods by 
executing a large number of independent, often small, jobs. This characteristic 
feature of HTC, frequently referred to as "task farming" poses problems for 
batch job schedulers such as Slurm used in HPC systems. A large number of jobs 
(launched with sbatch) and job steps (launched with srun) generate excess log 
data and slow down Slurm. Short jobs have also a large scheduling overhead, 
meaning that an increasing fraction of the time will actually be spent in the 
queue instead of computing. On an HPC system, you usually run work through a 
batch scheduler like SLURM. SLURM is great for large jobs, but it can be awkward 
when you have many smaller jobs. These constraints lead to some systems 
enforcing a limit on the number of srun invocations that you can perform on a 
given job submission

To get around this you would often submit a big job and then try to run many 
smaller tasks inside it. Jobs can finish at different times, leaving some CPU 
cores or GPUs idle even though there’s more work to do. Submitting a separate 
SLURM job for every task can be slow and creates scheduler overhead (lots of 
queueing, lots of launch latency) resulting in wasted compute time.

MatEnsemble
-----------

To solve this, MatEnsemble runs inside your allocated job and manages the work 
dynamically. It uses the Flux runtime to submit and track :obj:`Job`s as 
:obj:`Future`s.

At a high level, MatEnsemble repeatedly:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*       Checks what resources are free (CPU cores / GPUs).
*       Submits new tasks until resources are used up.
*       Processes completed tasks and records success/failure and outputs.
*       Repeats until everything is done. 

Logging
-------

Just as important as actually running the tasks is logging. MatEnsemble actively
logs your workflow status and organizes the output in an intuitive way. Each 
workflow creates a dedicated directory named "matensemble_workflow_YYYY-MM-DD_HH_mm_ss"

.. code-block:: 

   matensemble_workflow_1970-01-01_00-00-00/
   ├── status.json
   ├── matensemble_workflow.log
   └── out/
       ├── <job_id_1>/
       │   ├── stdout
       │   └── stderr
       ├── <job_id_2>/
       │   ├── stdout
       │   └── stderr
       └── .../          

The status.json file is used by the dashboard if it is enabled. 
Along side the status file there is a more verbose timestamped log file which 
holds a complete log of the workflow including any additional runtime information.

Each :obj:`Job` is given its own isolated directory for any output files that it 
produces. Along side the metadata that is produced by MatEnsemble there are the 
stdout and stderr files present in these directores that are used for any 
aditional information (E.g. error codes, print statements, stack traces, etc).

Adaptive Scheduling
-------------------

The key performance feature of MatEnsemble is adaptive scheduling.

In “adaptive” mode, the moment a :obj:`Job` finishes, MatEnsemble immediately submits 
a new one (if there’s remaining work and enough available resources). This keeps the 
worker pool saturated and the machine busy instead of waiting for a slow “batch” step. 

.. image:: ../../images/Cap_1_adaptive_task_management.png
   :alt: Adaptive Task Management

MatEnsemble Strategies
----------------------

To account for the different ways that a :obj:`Job` can be processed MatEnsemble 
uses the strategy pattern for processing the completion of :obj:`Job`s 

.. code-block:: python

    class FutureProcessingStrategy(ABC):
        @abstractmethod
        def process_futures(self, buffer_time) -> None:
            pass

The FutureProcessingStrategy is an interface that has one required method. Which
gives the manager more flexibility when processing tasks (i.e. adaptive or 
non-adaptive).

Out of the box MatEnsemble has two different strategies. 

FutureProcessingStrategy's
~~~~~~~~~~~~~~~~~~~~~~~~~~

*       **AdaptiveStrategy**
*       **NonAdaptiveStrategy**

Users can define their own strategies and provide them to MatEnsemble and 
the user defined strategies will be assigned automatically.

Conclusion 
----------

MatEnsemble is an adaptive workflow manager which is ideal for High Throughput 
workflows. It allows for full utilization of the resources that your given 
workflow has available. Making your :obj:`Job`s run more efficiently and stop
wasting precious compute. 

.. note::
   This project is under active development. Some APIs may change before 
   the 1.0 release.

