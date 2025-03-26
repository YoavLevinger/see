#!/bin/bash

echo "Starting task-splitter on port 8001..."
uvicorn task-splitter.app:app --port 8001 &

echo "Starting code-generator on port 8002..."
uvicorn code-generator.app:app --port 8002 &

echo "Starting tool-x-connector on port 8003..."
uvicorn tool-x-connector.app:app --port 8003 &

echo "Starting document-creator on port 8004..."
uvicorn document-creator.app:app --port 8004 &

echo "Starting main-controller on port 8080..."
uvicorn main-controller.main:app --port 8080 &

echo "All services are starting. Use Ctrl+C to stop them."
wait
