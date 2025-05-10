```python
import unittest
from refactored_module import FunctionA, FunctionB

class TestRefactoredCode(unittest.TestCase):
    def setUp(self):
        self.func_a = FunctionA()
        self.func_b = FunctionB()

    def test_function_a(self):
        self.assertEqual(self.func_a.process([1, 2, 3]), [2, 4, 8])
        self.assertEqual(self.func_a.process([]), [])
        self.assertEqual(self.func_a.process([0]), [0])

    def test_function_b(self):
        self.assertEqual(self.func_b.process('hello world'), 'olleh dlrow')
        self.assertEqual(self.func_b.process(''), '')
        self.assertEqual(self.func_b.process('xyz'), 'zyx')

if __name__ == '__main__':
    unittest.main()
```

Assuming that the `refactored_module.py` contains the following code:

```python
class FunctionA:
    def process(self, numbers):
        return [x**2 for x in numbers]

class FunctionB:
    def process(self, string):
        return string[::-1]
```