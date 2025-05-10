```python
def format_output(data):
    header = "{0:<20} {1:<10} {2:<10}\n".format("Name", "Age", "Score")
    lines = []
    for name, age, score in data:
        line = "{0:<20} {1:<10} {2:<10}\n".format(name, age, score)
        lines.append(line)
    return header + "".join(lines)
```

Assuming the input data is in a list of tuples:

```python
data = [("Alice", 25, 80), ("Bob", 30, 90), ("Charlie", 19, 75)]
```

You can call the function like this:

```python
print(format_output(data))
```