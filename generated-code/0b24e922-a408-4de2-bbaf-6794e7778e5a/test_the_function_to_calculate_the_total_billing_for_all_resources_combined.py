```python
class Resource:
    def __init__(self, price):
        self.price = price

    def get_billing(self):
        return self.price

def total_billing(resources):
    billing_sum = 0
    for resource in resources:
        billing_sum += resource.get_billing()
    return billing_sum

# Example usage:
resource1 = Resource(10)
resource2 = Resource(20)
all_resources = [resource1, resource2]
total_bill = total_billing(all_resources)
print(total_bill)  # Outputs: 30
```