# ./functions.py
from matensemble.pipeline import Pipeline

pipe = Pipeline()


@pipe.job()
def job1():
    return 1


@pipe.job()
def job2(x):
    return x + 1


@pipe.job()
def job3(x):
    return x * 2
