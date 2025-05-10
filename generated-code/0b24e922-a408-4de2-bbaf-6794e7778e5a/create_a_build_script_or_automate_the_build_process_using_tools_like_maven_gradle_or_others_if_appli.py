```python
import os
import shutil
from subprocess import call

# Define paths and project name
project_name = "my_project"
src_dir = f"src/{project_name}"
build_dir = f"{project_name}/build"
dist_dir = f"{project_name}/dist"

# Ensure directories are created
os.makedirs(build_dir, exist_ok=True)
os.makedirs(dist_dir, exist_ok=True)

# Copy source files to build directory
shutil.copytree(src_dir, build_dir)

# Run build command (replace with appropriate command for your tool)
call(["gradle", "build"], cwd=build_dir)

# Copy built files to distribution directory
shutil.copytree(f"{build_dir}/build/outputs/jar", dist_dir)
```

This code uses the `os`, `shutil`, and `subprocess` modules in Python 3. It assumes you are using Gradle as your build tool, but you can replace the "gradle build" command with the appropriate one for other tools like Maven or others.