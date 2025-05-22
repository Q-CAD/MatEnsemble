import concurrent.futures
import flux.job
import os
import flux
import pickle
import numpy as np
import os.path
import copy
import sys
import logging
from datetime import datetime
import time


__author__ = "Soumendu Bagchi"
__package__= 'matensemble'

class SuperFluxManager():

    def __init__(self, gen_task_list, \
                 gen_task_cmd, ml_task_cmd, \
                 ml_task_freq=100, write_restart_freq=100, \
                 tasks_per_job=None,
                 cores_per_task=1, gpus_per_task=0, \
                 cores_per_ml_task=1, restart_filename=None):

        self._running_tasks=[]
        self._completed_tasks=[]
        self._pending_tasks=copy.copy(gen_task_list)
        self._failed_tasks=[]
        self.flux_handle = flux.Flux()

        self.futures = set()
        self.tasks_per_job= copy.copy(list(tasks_per_job))
        self.cores_per_task = cores_per_task
        self.gpus_per_task = gpus_per_task
        self.cores_per_ml_task = cores_per_ml_task

        self.gen_task_cmd = gen_task_cmd
        self.ml_task_cmd = ml_task_cmd
        self.ml_task_freq = ml_task_freq
        self.write_restart_freq = write_restart_freq


        self.setup_logger()
        self.load_restart(restart_filename)

    @property
    def running_tasks(self):
        return self._running_tasks
    
    @property
    def completed_tasks(self):
        return self._completed_tasks
    
    @property
    def pending_tasks(self):
        return self._pending_tasks
    
    @property
    def failed_tasks(self):
        return self._failed_tasks
    
    # @property
    # def futures(self):
    #     return self._futures
    
    # @running_tasks.setter
    # def running_tasks(self, append_value=None, remove_value=None):
    #     if append_value != None:
    #         self._running_tasks.append(append_value)
    #     elif remove_value != None:
    #         self._running_tasks.remove(remove_value)
    #     else:
    #         pass

    # @completed_tasks.setter
    # def completed_tasks(self, append_value):
    #     self._completed_tasks.append(append_value)

    # @pending_tasks.setter
    # def pending_tasks(self, remove_ind):
    #     _ = self._pending_task.pop(remove_ind)

    # @failed_tasks.setter
    # def failed_tasks(self, append_value):
    #     self._failed_tasks.append_value(append_value)

    # @futures.setter
    # def futures(self, add_value):
    #     self._futures.add(add_value)

    def check_resources(self):

        self.status=flux.resource.status.ResourceStatusRPC(self.flux_handle).get()
        self.resource_list=flux.resource.list.resource_list(self.flux_handle).get()
        self.resource = flux.resource.list.resource_list(self.flux_handle).get()
        self.free_gpus = self.resource.free.ngpus
        self.free_cores = self.resource.free.ncores
        self.free_excess_cores = self.free_cores - self.free_gpus
        
    def update_resources(self, curr_num_task):

        self.free_excess_cores-=self.cores_per_task*curr_num_task
        self.free_cores-=self.cores_per_task*curr_num_task

        if self.gpus_per_task is not None:
            self.free_gpus-=self.gpus_per_task*curr_num_task


    def load_restart(self, filename):

        if (filename != None) and os.path.isfile(filename):
            try:
                self.completed_tasks, self.pending_tasks=pickle.load( open(filename, "rb" ) )
                self.logger.info("======WORKFLOW RESTARTING======")
                self.progress_monitor()
            except:
                pass


    def setup_logger(self):

        self.logger = logging.getLogger(__name__)

        stdoutHandler = logging.StreamHandler(stream=sys.stdout)
        date_time_hash = str(datetime.now())
        fileHandler = logging.FileHandler(f"logs-{date_time_hash}.txt")

        Fmt = logging.Formatter(
            "%(name)s %(asctime)s %(levelname)s %(filename)s %(lineno)s %(process)d %(message)s",
            defaults={"levelname": "severity", "asctime": "timestamp"},
            datefmt="%Y-%m-%dT%H:%M:%SZ",)

        stdoutHandler.setFormatter(Fmt)
        fileHandler.setFormatter(Fmt)

        self.logger.addHandler(stdoutHandler)
        self.logger.addHandler(fileHandler)

        self.logger.setLevel(logging.INFO)


    def progress_monitor(self, completed_tasks, running_tasks, pending_tasks):

        self.logger.info(f"======COMPLETED JOBS=======RUNNING JOBS======PENDING JOBS")
        self.logger.info(f"======{len(completed_tasks)}============{len(running_tasks)}============{len(pending_tasks)}")
        self.task_log = {'Completed tasks': completed_tasks, \
                    'Running tasks': running_tasks, \
                    'Pending tasks': pending_tasks}


    def process_futures(self, buffer_time):

        done, self.futures =concurrent.futures.wait(self.futures, timeout=buffer_time)
        for fut in done:
            self._completed_tasks.append(fut.task_)

            if len(self.completed_tasks)%self.ml_task_freq==0:
                # self.trigger_ml=True
                # print("Triggering ML workflow: ",len(self.completed_tasks))
                pickle.dump( (self.completed_tasks, self.pending_tasks) , open( f"restart_{len(self.completed_tasks)}.dat", "wb" ) )
            try:
                if fut.result() != 0:
                    self.logger.info(f"Task {fut.task_} exited with ERROR CODE {fut.result()}")
                self._running_tasks.remove(fut.task_)
            except:
                pass

            self.progress_monitor()


    
    def poolexecutor(self, task_arg_list, buffer_time=0.5, task_dir_list=None, adaptive=True):
        """ High-throughput executor implementation """

        # trigger_ml = False
        gen_task_arg_list = copy.copy(task_arg_list)
        gen_task_dir_list = copy.copy(task_dir_list)
        
        self.flux_handle.rpc("resource.drain", {"targets": "0"}).get()

        with flux.job.FluxExecutor() as executor:

          ### Queue Manager ###
            
            while True:

                """=================================================================================
                Stopping criteria 
                """
                if len(self.pending_tasks) == 0 and len(self.running_tasks) == 0:
                    self.logger.info("=============EXITING WORKFLOW ENVIRONMENT====================")
                    break
                """=================================================================================
                """


                #self.logger.info("=============INITIALIZING WORKFLOW ENVIRONMENT====================")

                if len(self.pending_tasks)>0:

                    self.check_resources()
                    
                    self.flux_handle.rpc("resource.drain", {"targets": "0"}).get()
                    if self.gpus_per_task>0:
                       
                        print ("resources----", "free cores: ", self.free_cores,"free gpus: ",self.free_gpus)
                        
                        #   FOR FRONTIER FREE CORES ARE EQUIV. TO FREE "EXCESS" CORES! SO CHANGING THE LOGIC BELOW ACCORDINGLY . . . 
                        while self.free_cores>=self.tasks_per_job[0]*self.cores_per_task and self.free_gpus>=self.tasks_per_job[0]*self.gpus_per_task and len(self.pending_tasks)>0:
                            
                            cur_task=self.pending_tasks[0]
                            cur_task_args = gen_task_arg_list[0]

                            _ = self._pending_tasks.pop(0)
                            _ = gen_task_arg_list.pop(0)

                            if gen_task_dir_list != None:
                                cur_task_dir = gen_task_dir_list[0]
                                _ = gen_task_dir_list.pop(0)
                            else:
                                cur_task_dir = None

                            flxt = Fluxlet(self.flux_handle, self.tasks_per_job[0], self.cores_per_task, self.gpus_per_task)
                            flxt.job_submit(executor, self.gen_task_cmd, cur_task, cur_task_args, cur_task_dir)

                            self.futures.add(flxt.future)
                            self._running_tasks.append(cur_task)
                            self.update_resources(int(self.tasks_per_job[0]))
                            print ("resources----", "free cores: ", self.free_cores,"free gpus: ",self.free_gpus)
                            _ = self.tasks_per_job.pop(0)
                            if len(self.tasks_per_job)==0:
                                self.tasks_per_job = [0]

                    
                    else:
                        print ("resources----", "free cores: ", self.free_cores,"free gpus: ",self.free_gpus)
                        while self.free_cores>=self.tasks_per_job[0]*self.cores_per_task and len(self.pending_tasks)>0:
                            
                            cur_task=self.pending_tasks[0]

                            cur_task_args = gen_task_arg_list[0]

                            _ = self._pending_tasks.pop(0)
                            _ = gen_task_arg_list.pop(0)

                            if gen_task_dir_list != None:
                                cur_task_dir = gen_task_dir_list[0]
                                _ = gen_task_dir_list.pop(0)
                            else:
                                cur_task_dir = None

                            flxt = Fluxlet(self.flux_handle, self.tasks_per_job[0], self.cores_per_task, self.gpus_per_task)
                            flxt.job_submit(executor, self.gen_task_cmd, cur_task, cur_task_args, cur_task_dir)

                            self.futures.add(flxt.future)
                            self._running_tasks.append(cur_task)
                            self.update_resources(int(self.tasks_per_job[0]))
                            print ("resources----", "free cores: ", self.free_cores,"free gpus: ",self.free_gpus)
                            _ = self.tasks_per_job.pop(0)
                            if len(self.tasks_per_job)==0:
                                self.tasks_per_job = [0]


                        #   TO BE IMPLEMENTED FOR THE ACTIVE LEARNING LOOP
                        # 
                        # if trigger_ml and self.free_excess_cores >=  self.cores_per_ml_task:
                             
                        #     """
                        #     implement ML Workflow HERE--also DATA EXTRACTION WORKLFLOW GOES HERE
                        #     """

                        #     trigger_ml=False

               # time.sleep(120)
                        # """

                # self.process_futures(buffer_time)

                done, self.futures =concurrent.futures.wait(self.futures, timeout=buffer_time) #return_when=concurrent.futures.FIRST_COMPLETED) #timeout=buffer_time)
                for fut in done:
                    self._completed_tasks.append(fut.task_)

                    ## Adaptive have to be implemented for heterogeneous task requirements

                    if len(self.completed_tasks)%self.write_restart_freq==0:
                        # self.trigger_ml=True
                        # print("Triggering ML workflow: ",len(self.completed_tasks))
                        self.task_log = {'Completed tasks': self.completed_tasks, \
                                         'Running tasks': self.running_tasks, \
                                         'Pending tasks': self.pending_tasks, \
                                         'Failed tasks': self.failed_tasks}
                        pickle.dump(self.task_log , open( f"restart_{len(self.completed_tasks)}.dat", "wb" ))
                    try:
                        print (f"Task {fut.task_} result: {fut.result(timeout=1200)}")
                        if fut.result(timeout=1200) != 0:
                            self.logger.info(f"Task {fut.task_} exited with ERROR CODE {fut.result()}")
                            self._failed_tasks.append(fut.task_)
                        self._running_tasks.remove(fut.task_)
                    except Exception as e:
                        print (f"Task {fut.task_}: could not process future.results() and exited with exception {e}") 
                        self._running_tasks.remove(fut.task_)

                    

                    self.logger.info(f"======COMPLETED JOBS=======RUNNING JOBS======PENDING JOBS")
                    self.logger.info(f"======{len(self.completed_tasks)}============{len(self.running_tasks)}============{len(self.pending_tasks)}")
            

class Fluxlet():

    def __init__(self, handle, tasks_per_job, cores_per_task, gpus_per_task):

        self.flux_handle = handle
        self.future = []
        self.tasks_per_job = tasks_per_job
        self.cores_per_task = cores_per_task
        self.gpus_per_task = gpus_per_task
        
    def job_submit(self, executor, command, task, task_args, task_directory=None):

        launch_dir = os.getcwd()
        cmd_list = [] #['flux', 'run', '-n', str(self.tasks_per_job), '-c', str(self.cores_per_task),'-g', str(self.gpus_per_task)]
        cmd_list.extend(command.split(" "))
#        cmd_list.append(os.path.abspath(command))
        
        if task_directory != None:
            
            try: 
                os.chdir(os.path.abspath(task_directory))
            except:
                msg = f"Could not find task directory {task_directory}: So, creating one instead . . ."
                print (msg)
                # self.logger.info
                os.mkdir(os.path.abspath(task_directory))
                os.chdir(task_directory)
        else:
            msg = "No directories are specified for the task. Task-list will serve as directory tree."
            print (msg)

            try: 
                os.chdir(str(task))
            except:
                os.mkdir(str(task))
                os.chdir(str(task))

        # print (task_args)
        # task_args = task_directory+'/'+task_args
        print (os.getcwd())
        if type(task_args) is list:
            str_args = [str(arg) for arg in task_args]
        elif type(task_args) is None:
            pass
        elif type(task_args) is str or type(task_args) is int or type(task_args) is float or type(task_args) is np.int64 or type(task_args) is np.float64 or type(task_args) is dict:
            str_args = [str(task_args)]
        else:
            raise(f"ERROR: Task argument can not be {type(task_args)}. Currently supports `list`, `str`, `int` and `float` types")

        
        cmd_list.extend(str_args)
#        print (cmd_list)

#        jobspec = flux.job.JobspecV1.from_nest_command(cmd_list,num_slots=self.tasks_per_job, \
#                                                       cores_per_slot=self.cores_per_task, \
#                                                       gpus_per_slot=self.gpus_per_task)
        
        #num_nodes=1,exclusive=True)
        print ('num_tasks:', self.tasks_per_job, " type of var :", type(self.tasks_per_job))
        jobspec = flux.job.JobspecV1.from_command(cmd_list,num_tasks=int(self.tasks_per_job), \
                                                       cores_per_task=self.cores_per_task, \
                                                       gpus_per_task=self.gpus_per_task) #, \
                                                       # num_nodes=1,exclusive=True)
  
        jobspec.cwd = os.getcwd()
        jobspec.setattr_shell_option("mpi","pmi2")
        jobspec.setattr_shell_option("cpu-affinity","per-task")
        jobspec.setattr_shell_option("gpu-affinity","per-task")
#        jobspec.setattr_shell_option("pmi","simple")
        jobspec.environment = dict(os.environ)
        jobspec.stdout = os.getcwd() + '/stdout'
        jobspec.stderr = os.getcwd() + '/stderr'

        self.resources = jobspec.resources

        self.future = executor.submit(jobspec)
        self.future.task_= task
        os.chdir(launch_dir)



            

