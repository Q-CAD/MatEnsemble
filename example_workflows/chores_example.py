from matensemble.pipeline import Pipeline

pipe = Pipeline()


@pipe.chore()
def factorial(n: int) -> int:
    """Caculate the factorial of a given integer"""

    product = 1
    for i in range(2, n):
        product *= i
    return product


@pipe.chore()
def digit_sum(n) -> int:
    """Cacluates the sum of each of the digits in a given integer"""

    sum = 0
    for char in str(n):
        sum += int(char)
    return sum


fact = factorial(100)
sum = digit_sum(fact)

future = pipe.submit(log_delay=1)
result = future.result()
print(result)
