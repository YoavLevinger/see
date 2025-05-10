```python
import sys
import os
from typing import Callable

class DataHandler:
    def __init__(self, load_data: Callable, save_data: Callable):
        self.load_data = load_data
        self.save_data = save_data

    def load(self, filepath: str) -> object:
        try:
            data = self.load_data(filepath)
            return data
        except Exception as e:
            print(f"Error occurred while loading data from {filepath}: {e}")
            sys.exit(1)

    def save(self, filepath: str, data: object):
        try:
            self.save_data(filepath, data)
        except Exception as e:
            print(f"Error occurred while saving data to {filepath}: {e}")
            sys.exit(1)
```

This code defines a `DataHandler` class that takes two functions (`load_data` and `save_data`) as arguments during initialization. The `load` method attempts to load data from a given file path, catching any exceptions and exiting the program if an error occurs. Similarly, the `save` method tries to save the provided data to a specified file path, also handling errors and terminating the program if necessary.