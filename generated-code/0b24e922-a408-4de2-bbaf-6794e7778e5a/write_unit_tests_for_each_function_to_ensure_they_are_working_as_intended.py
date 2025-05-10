```python
import unittest
from functions import function1, function2, function3

class TestFunctions(unittest.TestCase):
    def test_function1(self):
        self.assertEqual(function1([1, 2, 3]), [3, 2, 1])
        self.assertRaises(TypeError, function1, "string")

    def test_function2(self):
        self.assertEqual(function2("Hello"), "olleH")
        self.assertRaises(ValueError, function2, None)

    def test_function3(self):
        self.assertEqual(function3([1, 2, 3]), [True, False, True])
        self.assertRaises(TypeError, function3, "list")

if __name__ == "__main__":
    unittest.main()
```

In this code, I assumed you have a module named `functions.py` containing the functions: function1, function2, and function3. Replace the implementation of these functions according to your requirements. The test cases check if the functions return the expected results for valid inputs and if they raise appropriate exceptions for invalid inputs.