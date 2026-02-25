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
