```python
class Resource:
    def __init__(self, usage, rate):
        self.usage = usage
        self.rate = rate

class Billing:
    def __init__(self, resources):
        self.resources = resources

    def daily_billing(self):
        total_bill = 0
        for resource in self.resources:
            total_bill += resource.usage * resource.rate
        return total_bill

# Example usage
data = [Resource(10, 5), Resource(20, 7)]  # First element is usage, second is rate
billing_calculator = Billing(data)
print(billing_calculator.daily_billing())
```