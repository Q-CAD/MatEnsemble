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
