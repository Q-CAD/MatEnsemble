THIS IS AN EXAMPLE OF WHAT I WILL WANT USING MATENSEMBLE TO LOOK LIKE


Example problem that we are solving:

### Factorial Digit Sum

**n!** means **n x (n - 1) ... x 3 x 2 x 1**.

For example **10! = 10 x 9 ... x 2 x 1 = 3628800**,
and the sum of the digits of **10! is 3 + 6 + 2 + 8 + 8 + 0 + 0 = 27**.

Find the sum of the digits of **100!**.



```python

import matensemble.manager.SuperFluxManager
import matensemble.manager.Task
import matensemble.manager.SubTask


def factorial(n):
    product = 1
    for i in range(1, n + 1):
        product *= i
    return product

def main(n):
    s = f"{factorial(n)}"
    sum = 0
    for number in s:
        sum += int(number)
    print(f"The sum of the factorial digits in 100 is {sum}")



 ```
