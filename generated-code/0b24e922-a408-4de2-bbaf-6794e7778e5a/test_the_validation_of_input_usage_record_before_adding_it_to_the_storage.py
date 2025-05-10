```python
class UsageRecord:
    def __init__(self, user_id, operation, timestamp):
        self.user_id = int(user_id)
        self.operation = operation.lower()
        self.timestamp = timestamp

    @staticmethod
    def validate_input(user_id, operation, timestamp):
        if not isinstance(user_id, int) or user_id < 0:
            return False

        if len(operation) < 3 or len(operation) > 20:
            return False

        try:
            datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return False

        return True

    def add_to_storage(self):
        if self.validate_input(self.user_id, self.operation, self.timestamp):
            # Add to storage logic here
            pass
        else:
            raise ValueError("Invalid usage record")
```