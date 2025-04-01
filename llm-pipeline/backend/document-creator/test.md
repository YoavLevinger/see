curl -X POST http://localhost:8004/create \
  -H "Content-Type: application/json" \
  -d '{
    "folder_id": "test-folder",
    "description": "This is a test description",
    "subtasks": ["Subtask A", "Subtask B"],
    "dev_subtasks": ["Dev Subtask A", "Dev Subtask B"],
    "policy_texts": {
      "security": "Security policy content...",
      "accessibility": "Accessibility policy content..."
    },
    "expert_advice": {
      "advisor_1": "Advice text here..."
    },
    "effort_table": {
      "average_time": 123.45,
      "repositories": [
        {"name": "repo1", "hours": 100},
        {"name": "repo2", "hours": 146.9}
      ]
    }
  }'
