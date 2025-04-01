# shared/models.py

from pydantic import BaseModel
from typing import List, Dict, Optional, Any

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

# class DocRequest(BaseModel):
#     folder: str
#     description: str
#     subtasks: List[str]
#     dev_subtasks: List[str]

# class DocRequest(BaseModel):
#     folder_id: str
#     description: str
#     subtasks: List[str]
#     dev_subtasks: List[str]
#     policy_texts: Dict[str, str] = {}
#     expert_advice: Dict[str, str] = {}
#     effort_table: Optional[Dict[str, Any]] = {}

class DocRequest(BaseModel):
    folder_id: str
    description: str
    subtasks: List[str]
    dev_subtasks: List[str]
    policy_texts: Dict[str, str] = {}
    effort_table: Optional[Dict[str, Any]] = {}
    expert_advice: Dict[str, str] = {}

class AdvisorRequest(BaseModel):
    folder: str
    policy: str

class AdvisorResponse(BaseModel):
    recommendations: List[str]
