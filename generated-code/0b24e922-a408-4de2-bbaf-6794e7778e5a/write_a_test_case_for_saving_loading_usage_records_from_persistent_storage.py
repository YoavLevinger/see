```python
import unittest
from datetime import datetime
import json
from my_module import UsageRecord, Storage

class TestStorage(unittest.TestCase):
    def setUp(self):
        self.storage = Storage('test_data.json')
        self.record1 = UsageRecord(resource="CPU", usage=12345, timestamp=datetime(2022, 1, 1))
        self.record2 = UsageRecord(resource="RAM", usage=67890, timestamp=datetime(2022, 2, 2))

    def test_save(self):
        self.storage.save(self.record1)
        self.storage.save(self.record2)
        saved = self.storage.load()
        self.assertEqual(len(saved), 2)

    def test_load(self):
        self.storage.save(self.record1)
        self.storage.save(self.record2)
        loaded = self.storage.load()
        self.assertCountEqual([r.resource for r in loaded], ['CPU', 'RAM'])
        self.assertAlmostEqual([r.usage for r in loaded], [12345, 67890])
        self.assertEqual(len([r.timestamp for r in loaded]), 2)

    def tearDown(self):
        try:
            open('test_data.json', 'r').close()
        except FileNotFoundError:
            pass
        else:
            open('test_data.json', 'w').close()

if __name__ == "__main__":
    unittest.main()
```

Here, `my_module` is a hypothetical Python module that contains the `UsageRecord` and `Storage` classes with appropriate implementations for persistently saving/loading usage records in JSON format. The code above assumes that those classes have already been implemented.