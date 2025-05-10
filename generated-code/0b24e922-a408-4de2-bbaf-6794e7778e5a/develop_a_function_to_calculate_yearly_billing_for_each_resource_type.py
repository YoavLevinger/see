```python
def calculate_yearly_billing(resource_type, usage, price):
    def daily_billing(usage, price):
        return usage * price

    def yearly_billing(days_in_year=365):
        return daily_billing(usage, price) * days_in_year

    return {resource_type: yearly_billing}
```