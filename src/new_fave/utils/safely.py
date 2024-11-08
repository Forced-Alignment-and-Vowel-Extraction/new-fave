import warnings
from typing import Callable
def safely(
        message: str = f"There was a problem a function's application."
    ):
    """
    A decorator for more graceful failing. 
    If the decorated function raises an exception, 
    it will return `None`. 
    

    Args:
        message (str, optional): 
            A warning message in the case of an exception. 
            Defaults to `f"There was a problem a function's application."`.
    """
    def decorator(func:Callable):
        def safe_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                warnings.warn(message + "\n" + str(e))
                return None
        return safe_func
    return decorator