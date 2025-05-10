```python
def get_user_input():
    user_choice = input("Enter your choice (add, remove, view): ")
    if user_choice in ["add", "remove", "view"]:
        return user_choice
    else:
        print("Invalid choice. Please enter add, remove or view.")
        return get_user_input()

def business_logic(choice):
    if choice == "add":
        # Implement add logic here
    elif choice == "remove":
        # Implement remove logic here
    elif choice == "view":
        # Implement view logic here
```