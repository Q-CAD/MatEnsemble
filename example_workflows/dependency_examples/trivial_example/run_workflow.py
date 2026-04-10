# ./run_workflow.py
from functions import pipe, chore1, chore2, chore3

a = chore1()
b = chore2(a)
c = chore3(b)

pipe.submit()
