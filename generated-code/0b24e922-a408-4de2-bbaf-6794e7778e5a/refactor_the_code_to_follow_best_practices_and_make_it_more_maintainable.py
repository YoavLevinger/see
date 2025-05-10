```python
import re
from typing import List, Tuple

def parse_line(line: str) -> Tuple[str, int]:
    pattern = r'^(\w+) (\d+)$'
    match = re.match(pattern, line)
    if not match:
        raise ValueError("Invalid format")
    return match.groups()

def process_file(filename: str) -> List[str]:
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file]
        parsed_lines = [parse_line(line) for line in lines if line]
        return parsed_lines

def calculate_totals(data: List[Tuple[str, int]]) -> Tuple[int, int]:
    total_elf_calories = 0
    current_elf_calories = 0
    for name, calories in data:
        if calories == "":
            total_elf_calories += current_elf_calories
            current_elf_calories = 0
        else:
            current_elf_calories += int(calories)
    total_elf_calories += current_elf_calories
    return total_elf_calories, max(data, key=lambda x: x[1])[1]
```