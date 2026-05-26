from matensemble.pipeline import Pipeline

pipe = Pipeline()


# Define a chore that calculates the factorial of a given integer and another 
@pipe.chore()
def factorial(n: int) -> int:
    """Caculate the factorial of a given integer"""

    product = 1
    for i in range(2, n):
        product *= i
    return product


# Define a chore that calculates the sum of the digits in a given integer. 
@pipe.chore()
def digit_sum(n) -> int:
    """Cacluates the sum of each of the digits in a given integer"""

    sum = 0
    for char in str(n):
        sum += int(char)
    return sum


# We then use these two chores together to calculate the sum of the digits in 100!
fact = factorial(100)
sum = digit_sum(fact)

pipe.submit(log_delay=1)

# Print out the results of the workflow
print(pipe.results())
