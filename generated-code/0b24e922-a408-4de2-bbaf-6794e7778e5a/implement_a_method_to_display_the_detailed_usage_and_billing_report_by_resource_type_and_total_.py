```python
from collections import defaultdict

class ResourceUsageReport:
    def __init__(self):
        self.usage_report = defaultdict(lambda: 0)
        self.billing_report = defaultdict(float)

    def log_usage(self, resource_type, hours_used):
        self.usage_report[resource_type] += hours_used

    def log_bill(self, resource_type, cost_per_hour):
        self.billing_report[resource_type] += cost_per_hour * self.usage_report[resource_type]

    def display_report(self):
        print("\nDetailed Usage and Billing Report")
        print("----------------------------------")
        for resource_type, usage in sorted(self.usage_report.items()):
            bill = self.billing_report[resource_type]
            print(f"{resource_type}: {usage} hours used ({'$' + str(bill)})")
```