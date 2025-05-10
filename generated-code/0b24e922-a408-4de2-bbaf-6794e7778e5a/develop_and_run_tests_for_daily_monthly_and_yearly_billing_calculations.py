```python
import unittest
from datetime import timedelta, date

class BillingTest(unittest.TestCase):
    def setUp(self):
        self.billing_rate = 10.0
        self.base_day_duration = timedelta(hours=24)

    def daily_billing(self, start_date, end_date):
        days = (end_date - start_date).days + 1
        return days * self.billing_rate

    def monthly_billing(self, start_date, end_date):
        first_day = date(start_date.year, start_date.month, 1)
        last_day = end_date if (end_date.month == start_date.month and end_date.day >= start_date.day) else date(start_date.year, end_date.month, 1).replace(day=end_date.day)
        days_in_month = (last_day - first_day).days + 1
        return self.daily_billing(first_day, last_day) * days_in_month * self.billing_rate

    def yearly_billing(self, start_date, end_date):
        start_year = start_date.year
        end_year = end_date.year if end_date > start_date else start_date.year + 1
        return sum([self.monthly_billing(date(start_year, month=i+1, day=1), date(end_year, 12, 31)) for i in range(12)])

    def test_daily_billing(self):
        start_date = date(2022, 1, 1)
        end_date = date(2022, 1, 5)
        expected = 5 * self.billing_rate
        self.assertEqual(self.daily_billing(start_date, end_date), expected)

    def test_monthly_billing(self):
        start_date = date(2022, 1, 1)
        end_date = date(2022, 2, 1)
        expected = (date(2022, 2, 1) - date(2022, 1, 1)).days * self.billing_rate * 31
        self.assertEqual(self.monthly_billing(start_date, end_date), expected)

    def test_yearly_billing(self):
        start_date = date(2021, 10, 5)
        end_date = date(2022, 9, 30)
        expected = sum([self.monthly_billing(date(2021, month=i+1, day=1), date(2021, 12, 31)) for i in range(6)] + [self.monthly_billing(date(2022, month, day=1), date(2022, 9, 30)) for month in range(1, 10)])
        self.assertEqual(self.yearly_billing(start_date, end_date), expected)

if __name__ == '__main__':
    unittest.main()
```