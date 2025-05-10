```python
import json

def save_usage_records(records, filename='usage_records.json'):
    try:
        with open(filename, 'w') as f:
            json.dump(records, f)
    except Exception as e:
        print(f"Error saving usage records: {e}")

def load_usage_records(filename='usage_records.json'):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
```