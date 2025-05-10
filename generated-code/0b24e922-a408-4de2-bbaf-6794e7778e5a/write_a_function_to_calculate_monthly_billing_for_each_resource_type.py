```python
def calculate_monthly_billing(usage):
    resource_rates = {
        "data": 0.1,
        "computation": 0.075,
        "storage": 0.03
    }

    billing = {}

    for resource_type, usage_amount in usage.items():
        billing[resource_type] = resource_rates.get(resource_type, 0) * usage_amount

    return billing
```