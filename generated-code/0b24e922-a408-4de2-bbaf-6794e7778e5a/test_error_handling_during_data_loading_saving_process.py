```python
import csv

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        try:
            with open(self.file_path, 'r') as file:
                reader = csv.reader(file)
                data = list(reader)
            return data
        except FileNotFoundError:
            print("File not found")
        except Exception as e:
            print(f"Unexpected error occurred during loading data: {e}")

    def save(self, data):
        try:
            with open(self.file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(data)
        except FileNotFoundError:
            print("File not found")
        except Exception as e:
            print(f"Unexpected error occurred during saving data: {e}")
```