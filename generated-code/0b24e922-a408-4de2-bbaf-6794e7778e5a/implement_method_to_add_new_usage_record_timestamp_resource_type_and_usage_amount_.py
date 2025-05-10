```python
class UsageRecord:
    def __init__(self, timestamp, resource_type, usage_amount):
        self.timestamp = timestamp
        self.resource_type = resource_type
        self.usage_amount = usage_amount

def add_usage_record(records, timestamp, resource_type, usage_amount):
    new_record = UsageRecord(timestamp, resource_type, usage_amount)
    records.append(new_record)
    return records
```

You can use this class and method to create a new UsageRecord object with given timestamp, resource_type, and usage_amount, and add it to the list of records. This implementation assumes that you have an initial empty list to store the usage records. For example:

```python
records = []
add_usage_record(records, "2022-01-01 10:00:00", "CPU", 50)
add_usage_record(records, "2022-01-01 11:00:00", "RAM", 100)
```