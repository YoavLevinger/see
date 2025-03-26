from pydantic import BaseModel
from typing import List

class TaskRequest(BaseModel):
    description: str

class SubTaskResponse(BaseModel):
    subtasks: List[str]

class CodeGenRequest(BaseModel):
    subtask: str
    folder: str

class ToolXRequest(BaseModel):
    folder: str
    description: str
    subtasks: List[str]

class DocRequest(BaseModel):
    folder: str
    description: str
    subtasks: List[str]
