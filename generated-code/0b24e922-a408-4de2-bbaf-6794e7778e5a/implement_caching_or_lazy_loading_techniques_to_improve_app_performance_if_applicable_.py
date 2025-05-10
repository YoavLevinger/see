```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(arg):
    # Your expensive function implementation here
    pass

# Usage of the lazy-loaded function
expensive_function('some_arg')
```

In this example, `expensive_function` uses the LRU (Least Recently Used) cache from the `functools` module. This means that the most recently used cache entries are kept as long as possible and the least recently used entries are removed first when the cache exceeds its maximum size. The maximum size is set to 128 in this example, but it can be adjusted according to your needs.