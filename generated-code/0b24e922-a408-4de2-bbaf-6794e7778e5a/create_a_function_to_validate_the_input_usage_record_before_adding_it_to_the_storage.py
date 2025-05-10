```python
def validate_usage(record):
    try:
        device_id = int(record['device_id'])
        timestamp = float(record['timestamp'])
        data_consumed = float(record['data_consumed'])

        if 0 <= device_id < sys.maxsize and timestamp >= 0 and data_consumed > 0:
            return True
    except (KeyError, ValueError):
        return False
    return False
```
In this code, the validate_usage function takes a dictionary as an argument that contains 'device_id', 'timestamp' and 'data_consumed'. It checks if the values are integers for device_id and floats for timestamp and data_consumed. Additionally, it verifies that the values meet certain conditions (non-negative for device_id and data_consumed > 0). If all checks pass, it returns True indicating valid input; otherwise False.