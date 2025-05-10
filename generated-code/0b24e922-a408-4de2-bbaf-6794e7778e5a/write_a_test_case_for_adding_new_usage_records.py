```python
import unittest
from your_module import UsageRecord, RecordsCollection

class TestUsageRecords(unittest.TestCase):
    def setUp(self):
        self.records = RecordsCollection()

    def test_add_new_usage_record(self):
        record = UsageRecord("2022-12-31", "device1", 5)
        self.records.add_record(record)
        self.assertEqual(len(self.records._records), 1)
        self.assertEqual(self.records.get_usage_record_by_date("2022-12-31"), record)

if __name__ == '__main__':
    unittest.main()
```