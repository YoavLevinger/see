```python
def get_valid_input(prompt):
    while True:
        try:
            user_input = input(prompt)
            if user_input.isnumeric():
                return int(user_input)
            else:
                print("Invalid input. Please enter a number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
```
This function will keep prompting the user for an input until they provide a valid numeric input. It uses a try-except block to handle invalid inputs and provide an error message.