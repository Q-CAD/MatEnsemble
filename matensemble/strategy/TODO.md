
Here are the things that you need to do.

So basically there are two main strategies and you need to look through the code to see what the common patterns between those two strategies are. They basically look like this 

* check resources 
* while you have enough resources to submit a task do that 
    * update resources and log them
    * submit a task with Fluxlet
    * update task list 
    * add the job to the futures list
    * update resources and sleep for a buffer_time
* once you are out of resources... continue super loop


This happens inside of the super loop which goes like this

* while you still have pending tasks and you still have running tasks
    * implement the submission strategy 
    * process futures 
    * update lists (futures_list, running_tasks, completed_tasks, pending_tasks, failed_tasks)

    * continue super loop 
