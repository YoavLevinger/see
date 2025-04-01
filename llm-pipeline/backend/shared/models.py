# shared/models.py

from pydantic import BaseModel
from typing import List

class TaskRequest(BaseModel):
    description: str

class SubTaskResponse(BaseModel):
    subtasks: List[str]              # full list: planning, testing, etc.
    dev_subtasks: List[str]          # development-only (for code gen)

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
    dev_subtasks: List[str]

class AdvisorRequest(BaseModel):
    folder: str
    policy: str

class AdvisorResponse(BaseModel):
    recommendations: List[str]
