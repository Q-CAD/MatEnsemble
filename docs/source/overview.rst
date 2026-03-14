==============
Overview Hello
==============

MatEnsemble is a workflow manager for running lots of similar simulation tasks 
on a supercomputer as efficiently as possible. MatEnsemble launches tasks,
watches them finish, records what happened, and immediately sends more when 
resources open up. 

High Throughput Computing
-------------------------
High Throughput Computing (HTC) is a computing paradigm designed to maximize 
the total amount of computational work completed over long periods by 
executing a large number of independent, often small, tasks. This characteristic 
feature of HTC, frequently referred to as "task farming" poses problems for 
batch job schedulers such as Slurm used in HPC systems. A large number of jobs 
(launched with sbatch) and job steps (launched with srun) generate excess log 
data and slow down Slurm. Short jobs have also a large scheduling overhead, 
meaning that an increasing fraction of the time will actually be spent in the 
queue instead of computing. On an HPC system, you usually run work through a 
batch scheduler like SLURM. SLURM is great for large jobs, but it can be awkward 
when you have many smaller tasks. These constraints lead to some systems 
enforcing a limit on the number of srun invocations that you can perform on a 
given job submission

To get around this you would often submit a big “job” and then try to run many 
smaller tasks inside it. Tasks can finish at different times, leaving some CPU 
cores or GPUs idle even though there’s more work to do. Submitting a separate 
SLURM job for every task can be slow and creates scheduler overhead (lots of 
queueing, lots of launch latency) resulting in wasted compute time.

MatEnsemble
-----------

To solve this, MatEnsemble runs inside your allocated job and manages the work 
dynamically. It uses the Flux runtime to submit and track tasks as “futures,” 
like a fast in-job task queue. 

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

.. code-block:: bash

   matensemble_workflow_1970-01-01_00-00-00/
   ├── status.log
   ├── matensemble_workflow.log
   └── out/
       └── <output_of_workflow>

The status.log file is a minimal text file that is updated in place and can be 
watched to check the progress of your workflow.

.. code-block:: bash

   watch -n 1 cat <workflow_dir>/status.log

Along side the status file there is a more verbose timestamped log file which 
holds a complete log of the workflow including any additional runtime information.

Each task is given its own isolated directory for any output files that it 
produces. Also present in these directories is a stdout and stderr for any
aditional information including error codes if a job fails. 


Adaptive Scheduling
-------------------

The key performance feature is adaptive task scheduling.

In “adaptive” mode, the moment a task finishes, MatEnsemble immediately submits 
a new one (if there’s remaining work and available resources). This keeps the 
worker pool saturated and the machine busy instead of waiting for a slow “batch” step. 

.. image:: ../../images/Cap_1_adaptive_task_management.png
   :alt: Adaptive Task Management


MatEnsemble Strategies
----------------------

Tasks can have many different *flavors*. Some tasks can only be run on the CPU 
while others have an affinity towards the GPU. 

To account for these differences MatEnsemble uses the 'Strategy Pattern' to make 
the manager more modular, customizable and maintainable. Right now there are
two different strategies that MatEnsemble depends on. The TaskSubmissionStrategy
and the FutureProcessingStrategy.

.. code-block:: python

    class TaskSubmissionStrategy(ABC):
        @abstractmethod
        def submit_until_ooresources(
            self, task_arg_list, task_dir_list, buffer_time
        ) -> None:
            pass

        @abstractmethod
        def submit(
            self, task, tasks_per_job, task_args, task_dir
        ) -> flux.job.executor.FluxExecutorFuture:
            pass

The TaskSubmissionStrategy is an interface that has two required methods. This
gives the manager the flexibility to treat different tasks according to their 
preferences (i.e. GPU affinity or CPU affinity).

.. code-block:: python

    class FutureProcessingStrategy(ABC):
        @abstractmethod
        def process_futures(self, buffer_time) -> None:
            pass

The FutureProcessingStrategy is an interface that has one required method. Which
gives the manager more flexibility when processing tasks (i.e. adaptive or 
non-adaptive).

Out of the box MatEnsemble has five different strategies. Three TaskSubmissionStrategy's
and two FutureProcessingStrategy's. 

TaskSubmissionStrategy's
~~~~~~~~~~~~~~~~~~~~~~~~

*       **CPUAffineStrategy**
*       **GPUAffineStrategy**
*       **DynoproStrategy**

FutureProcessingStrategy's
~~~~~~~~~~~~~~~~~~~~~~~~~~

*       **AdaptiveStrategy**
*       **NonAdaptiveStrategy**

Users can also define their own strategies and provide them to MatEnsemble and 
the user defined strategies will be assigned automatically.

Conclusion 
----------

MatEnsemble is an adaptive task manager which is ideal for High Throughput 
workflows, and allows for full utilization of the resources that your given 
workflow has available. Making your jobs run more efficiently and stop
wasting precious compute. 

.. note::
   This project is under active development. Some APIs may change before 
   the 1.0 release.

