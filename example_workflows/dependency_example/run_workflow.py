# ./run_workflow.py
from functions import pipe, job1, job2, job3

a = job1()
b = job2(a)
c = job3(b)

pipe.submit()
