```python
import requests
import json
from time import sleep

STAGING_API_URL = "https://your-staging-app-url.com"
ISSUE_REPORT_URL = STAGING_API_URL + "/report_issue"

def check_api():
    response = requests.get(STAGING_API_URL)
    if response.status_code != 200:
        report_issue("API returned non-200 status code")

def run_test_case(test_case):
    response = requests.post(STAGING_API_URL + "/run", json=test_case)
    if response.status_code != 200:
        report_issue(f"Test case {test_case['name']} failed")

def report_issue(message):
    data = {"message": message}
    requests.post(ISSUE_REPORT_URL, json=data)

def main():
    test_cases = [
        {"name": "Test case 1", "parameters": {}},
        # Add more test cases here...
    ]

    for test_case in test_cases:
        run_test_case(test_case)

    check_api()

if __name__ == "__main__":
    main()
```