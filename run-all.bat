@echo off
REM === Kill processes on ports if already running ===
FOR %%P IN (8001 8002 8003 8004 8005 8006 8007 8080 8090) DO (
    FOR /F "tokens=5" %%A IN ('netstat -ano ^| findstr :%%P') DO (
        taskkill /PID %%A /F >nul 2>&1
    )
)
REM === Set PYTHONPATH ===
set PYTHONPATH=%CD%

REM === Start backend services ===
start "Task Splitter" cmd /k python -m uvicorn backend.task-splitter.task_splitter:app --port 8001
start "Code Generator" cmd /k python -m uvicorn backend.code-generator.code_generator:app --port 8002 --workers 16
start "Tool X Connector" cmd /k python -m uvicorn backend.tool-x-connector.tool_x_connector:app --port 8003
start "Document Creator" cmd /k python -m uvicorn backend.document-creator.document_creator:app --port 8004
start "Expert Advisor" cmd /k python -m uvicorn backend.expert-advisor.expert_advisor:app --port 8005
start "SBERT Estimator" cmd /k python -m uvicorn backend.sbert_complexity_estimator.effort_estimator_combined:app --port 8007
start "Main Controller" cmd /k python -m uvicorn backend.main-controller.main_controller:app --port 8080
start "Frontend App" cmd /k python -m uvicorn frontend.frontend_app:app --port 8090

echo ✅ All services started. Press Ctrl+C in each window to stop.
