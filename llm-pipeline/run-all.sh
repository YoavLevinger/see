#!/bin/bash


# Kill any leftover processes from last run
for port in 8001 8002 8003 8004 8005 8006 8007 8080 8090; do
  pid=$(lsof -t -i :$port)
  if [ -n "$pid" ]; then
    echo "🔴 Killing process on port $port (PID: $pid)"
    kill -9 $pid
  fi
done


# Navigate to the project root
cd "$(dirname "$0")"

# Set PYTHONPATH to the project root so imports like `from shared.models` work
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Starting task-splitter on port 8001..."
uvicorn backend.task-splitter.task_splitter:app --port 8001 &

echo "Starting code-generator on port 8002..."
uvicorn backend.code-generator.code_generator:app --port 8002 --workers 16 &

echo "Starting tool-x-connector on port 8003..."
uvicorn backend.tool-x-connector.tool_x_connector:app --port 8003 &

echo "Starting document-creator on port 8004..."
uvicorn backend.document-creator.document_creator:app --port 8004 &

echo "Starting expert-advisor on port 8005..."
uvicorn backend.expert-advisor.expert_advisor:app --port 8005 &

echo "Starting sbert-complexity-estimator on port 8006..."
#uvicorn backend.sbert-complexity-estimator.sbert_complexity_estimator:app --port 8006 &
uvicorn backend.sbert_complexity_estimator.effort_estimator_combined:app --port 8007 &

echo "Starting main-controller on port 8080..."
uvicorn backend.main-controller.main_controller:app --port 8080 &

echo "Starting frontend on port 8090..."
uvicorn frontend.frontend_app:app --port 8090 &

echo "✅ All services are starting. Press Ctrl+C to stop them."
wait
