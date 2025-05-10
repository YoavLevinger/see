```python
import tkinter as tk
from tkinter import messagebox

class UsageRecordUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Usage Record UI")
        self.frame = tk.Frame(self.root)
        self.frame.pack()

        self.entry_id = tk.Entry(self.frame, width=20)
        self.entry_id.grid(row=0, column=0, padx=10, pady=10)

        self.entry_date = tk.Entry(self.frame, width=20)
        self.entry_date.grid(row=1, column=0, padx=10, pady=10)

        self.entry_usage = tk.Entry(self.frame, width=20)
        self.entry_usage.grid(row=2, column=0, padx=10, pady=10)

        self.btn_add = tk.Button(self.frame, text="Add Record", command=self.add_record)
        self.btn_add.grid(row=3, column=0, padx=10, pady=10)

        self.listbox = tk.Listbox(self.frame, width=50, height=10)
        self.listbox.grid(row=4, column=0, rowspan=5, padx=10, pady=10)

    def add_record(self):
        id = self.entry_id.get()
        date = self.entry_date.get()
        usage = self.entry_usage.get()

        if not id or not date or not usage:
            messagebox.showerror("Error", "All fields are required.")
            return

        self.listbox.insert(tk.END, f"ID: {id}, Date: {date}, Usage: {usage}")
        self.clear_entry()

    def clear_entry(self):
        self.entry_id.delete(0, tk.END)
        self.entry_date.delete(0, tk.END)
        self.entry_usage.delete(0, tk.END)

root = tk.Tk()
UsageRecordUI(root)
root.mainloop()
```
This code creates a simple Tkinter GUI for entering and viewing usage records. The user can input an ID, date, and usage in the text boxes, then click "Add Record" to add the record to the listbox below. The entries are automatically cleared after adding a record.