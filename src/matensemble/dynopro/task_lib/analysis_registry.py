from functools import wraps
from typing import Callable

class AnalysisRegistry:


    """
    # Example of adding a new custom analysis
    @AnalysisRegistry.register('my_custom_analysis')
    def my_custom_analysis(data, params):
        # Your custom analysis code here
        result = some_calculation(data, **params)
        
        # Save results
        with open(f'custom_analysis_{data.timestep}', 'w') as file:
            file.write(f'time-step result\n')
            file.write(f'{data.timestep} {result}')
    """
    _registry = {}

    @classmethod
    def register(cls, name: str):
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            cls._registry[name] = wrapper
            return wrapper
        return decorator

    @classmethod
    def get_registered_analyses(cls):
        return cls._registry