```python
import pickle
import os

def load_usage_records(filepath):
    if not os.path.exists(filepath):
        return []

    with open(filepath, 'rb') as f:
        usage_records = pickle.load(f)

    return usage_records
```