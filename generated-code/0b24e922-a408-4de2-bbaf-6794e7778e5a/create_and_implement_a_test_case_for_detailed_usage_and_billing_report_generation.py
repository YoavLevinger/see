```python
import unittest
from datetime import datetime, timedelta

class TestDetailedUsageReport(unittest.TestCase):
    def setUp(self):
        self.start_date = datetime(2021, 1, 1)
        self.end_date = datetime(2021, 1, 7)
        self.usage_data = [
            ("John Doe", "api_calls", 500, self.start_date),
            ("Jane Smith", "api_calls", 800, self.start_date + timedelta(days=1)),
            ("John Doe", "storage", 100, self.start_date + timedelta(days=2)),
            ("John Doe", "api_calls", 300, self.start_date + timedelta(days=3)),
            ("Jane Smith", "storage", 250, self.end_date),
        ]

    def test_generate_report(self):
        from report import DetailedUsageReport
        report = DetailedUsageReport(self.start_date, self.end_date)
        report.add_usage(self.usage_data)
        expected_output = [
            ("John Doe", "api_calls", 800, self.start_date),
            ("Jane Smith", "api_calls", 0, self.start_date + timedelta(days=1)),
            ("John Doe", "storage", 100, self.start_date + timedelta(days=2)),
            ("Jane Smith", "storage", 250, self.end_date),
        ]
        self.assertEqual(report.generate(), expected_output)

class TestBillingReport(unittest.TestCase):
    def setUp(self):
        self.start_date = datetime(2021, 1, 1)
        self.end_date = datetime(2021, 1, 7)
        self.usage_data = [
            ("John Doe", "api_calls", 500, self.start_date),
            ("Jane Smith", "api_calls", 800, self.start_date + timedelta(days=1)),
            ("John Doe", "storage", 100, self.start_date + timedelta(days=2)),
            ("John Doe", "api_calls", 300, self.start_date + timedelta(days=3)),
            ("Jane Smith", "storage", 250, self.end_date),
        ]

    def test_generate_report(self):
        from report import BillingReport
        rate = {
            ("John Doe", "api_calls"): 0.1,
            ("Jane Smith", "api_calls"): 0.2,
            ("John Doe", "storage"): 5,
            ("Jane Smith", "storage"): 4,
        }
        report = BillingReport(self.start_date, self.end_date, rate)
        report.add_usage(self.usage_data)
        expected_output = [
            ("John Doe", "api_calls", 80, self.start_date),
            ("Jane Smith", "api_calls", 160, self.start_date + timedelta(days=1)),
            ("John Doe", "storage", 500, self.start_date + timedelta(days=2)),
            ("Jane Smith", "storage", 1000, self.end_date),
        ]
        self.assertEqual(report.generate(), expected_output)
```

Assuming the `report.py` file contains the following code:

```python
class DetailedUsageReport():
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.data = []

    def add_usage(self, usage_data):
        for item in usage_data:
            self.data.append((item[0], item[1], item[2], max(self.start_date, item[3])))

    def generate(self):
        return sorted(self.data, key=lambda x: x[3])

class BillingReport():
    def __init__(self, start_date, end_date, rate):
        self.start_date = start_date
        self.end_date = end_date
        self.rate = rate
        self.data = []

    def add_usage(self, usage_data):
        for item in usage_data:
            self.data.append((item[0], item[1], item[2] * self.rate[(item[0], item[1])], max(self.start_date, item[3])))

    def generate(self):
        return sorted(self.data, key=lambda x: x[3])
```