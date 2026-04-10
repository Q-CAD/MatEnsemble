# fibonacci_chain_workflow.py
from matensemble.pipeline import Pipeline

pipe = Pipeline()


@pipe.chore(name="fib-seed")
def fib_seed(value: int) -> int:
    return value


@pipe.chore(name="fib-step")
def fib_step(a: int, b: int, n: int) -> int:
    value = a + b
    print(f"Computed F({n}) = {value}")
    return value


@pipe.chore(name="fib-report")
def fib_report(n: int, value: int) -> dict:
    result = {"n": n, "value": value}
    print(result)
    return result


def build_workflow(n: int = 10):
    if n < 0:
        raise ValueError("n must be >= 0")

    if n == 0:
        out = fib_seed(0)
        fib_report(0, out)
        return

    f0 = fib_seed(0)
    f1 = fib_seed(1)

    if n == 1:
        fib_report(1, f1)
        return

    prev2 = f0
    prev1 = f1

    for i in range(2, n + 1):
        current = fib_step(prev2, prev1, i)
        prev2, prev1 = prev1, current

    fib_report(n, prev1)
