```python
class Resource:
    def __init__(self, cost):
        self.cost = cost

    def get_cost(self):
        return self.cost

class Bill:
    def __init__(self, resources):
        self.resources = resources

    def calculate_total(self):
        total = 0
        for resource in self.resources:
            total += resource.get_cost()
        return total
```

Example usage:

```python
resource1 = Resource(10)
resource2 = Resource(5)
bill = Bill([resource1, resource2])
total = bill.calculate_total()
print(total)  # Outputs: 15
```