```python


# parse args from spec. The args should contain the module where the function 
# is located and the arguments for the function call itself.

def main():
    module = args[name of arg]
    function_name = args[name of function]
    from module import function_name

    result = function_name(args)
    # store result either in KVS or in a file 
    write the file or kvs add (...) + kvs commit()


if __name__ == "__main__":
    main(args, kwargs)


```
