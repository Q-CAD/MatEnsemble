# ./functions.py
from matensemble.pipeline import Pipeline

pipe = Pipeline()


@pipe.chore()
def chore1():
    return 1


@pipe.chore()
def chore2(x):
    return x + 1


@pipe.chore()
def chore3(x):
    return x * 2


# ./run_workflow.py
from functions import pipe, chore1, chore2, chore3

a = chore1()
b = chore2(a)
c = chore3(b)

pipe.submit()
