```python
import unittest
from pathlib import Path
from your_build_script import BuildScript

class TestBuildScript(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).parent / 'tests'
        self.bs = BuildScript(self.test_dir)

    def test_build_with_default_config(self):
        output = self.bs.run()
        # Adjust this assertion according to the expected behavior of your build script
        self.assertTrue(output, msg="Build failed with default config")

    def test_build_with_custom_config(self):
        custom_config_path = self.test_dir / 'custom_config.yaml'
        output = self.bs.run_with_config(custom_config_path)
        # Adjust this assertion according to the expected behavior of your build script
        self.assertTrue(output, msg="Build failed with custom config")

if __name__ == '__main__':
    unittest.main()
```

This code provides a basic test structure for testing your build script using Python's built-in `unittest` module. It assumes that you have created a YAML configuration file (custom_config.yaml) in the 'tests' folder to test with custom configurations. Adjust the tests and assertions according to the expected behavior of your build script.