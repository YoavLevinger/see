```python
class ResourceUsageTracker:
    def __init__(self):
        self.usage = {}

    def add_usage(self, resource_type, usage_amount):
        if resource_type not in self.usage:
            self.usage[resource_type] = 0
        self.usage[resource_type] += usage_amount

    def get_total_usage(self, resource_type):
        if resource_type not in self.usage:
            return None
        return self.usage[resource_type]
```

This code defines a class `ResourceUsageTracker` that tracks the total usage of different resources. You can create an instance of this class and use the `add_usage()` method to add usage for a specific resource type, and `get_total_usage()` method to retrieve the total usage for a given resource type. If the resource type does not exist in the tracker, the `get_total_usage()` method will return `None`.