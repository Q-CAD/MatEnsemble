THIS IS AN EXAMPLE OF WHAT I WILL WANT USING MATENSEMBLE TO LOOK LIKE


Example problem that we are solving:

### Factorial Digit Sum

**n!** means **n x (n - 1) ... x 3 x 2 x 1**.

For example **10! = 10 x 9 ... x 2 x 1 = 3628800**,
and the sum of the digits of **10! is 3 + 6 + 2 + 8 + 8 + 0 + 0 = 27**.

Find the sum of the digits of **100!**.



--------------------------------------------------------------------

```python
import # imports here

pipline = Pipeline()

# @decorator here maybe 
@pipeline.task() # or .job() idk 
def factorial(n):
    product = 1
    for i in range(1, n + 1):
        product *= i
    return product

@pipeline.task() # or .job(depends on factorial)
def factorial_digit_sum(n):
    s = f"{factorial(n)}"
    sum = 0
    for number in s:
        sum += int(number)
    print(f"The sum of the factorial digits in 100 is {sum}")


# then how would they put thier arguments to start. 

first = factorial(100)
second = factorial_digit_sum(first.output)


# MatEnsemble API
pipeline.run(second)


--------------------------------------------------------------------


# THis is different from Jobflow's API they have it like this

from jobflow import job

@job
def add(a, b):
    return a + b

add_first = add(1, 5)
add_second = add(add_first.output, 3)

from jobflow import Flow

flow = Flow([add_first, add_second])
flow.draw_graph(figsize=(3, 3)).show()



 ```

In the future we also should be able to use a tool like 
graphviz to show a visual of the workflow. 


```python
import matensemble.pipeline

pipeline = Pipeline()

@pipeline.task()
def factorial(n):
    product = 1
    for i in range(1, n+1):
        product *= i
    return product

@pipeline.task()
def digit_sum(n):
    return sum(int(d) for d in str(n))

n = factorial(100)
sum = digit_sum(n)
result = pipeline.run(sum)
print(result)
```
