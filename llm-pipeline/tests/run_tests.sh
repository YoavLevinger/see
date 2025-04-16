#!/bin/bash

# Endpoint
URL="http://localhost:8080/process"

# List of 32 software task descriptions
DESCRIPTIONS=(
  "Build a REST API for a to-do list application"
  "Create a real-time chat application using websockets"
  "Develop a single-page app for online shopping"
  "Implement a file upload and processing backend"
  "Build a dashboard for IoT sensor data monitoring"
  "Create an OAuth2 login system with Google and GitHub"
  "Develop a CI/CD pipeline using GitHub Actions"
  "Create an admin panel for a blogging platform"
  "Implement a PDF report generator for time logs"
  "Build a static site generator in Python"
  "Develop an e-commerce shopping cart with checkout"
  "Create a scheduler for employee shifts"
  "Build a Slack bot for task reminders"
  "Develop a system for image classification using CNN"
  "Create an API gateway with rate limiting"
  "Implement a job queue and worker system"
  "Build a microservice for currency conversion"
  "Create a service that scrapes news headlines daily"
  "Develop a webhook listener for payment notifications"
  "Create a CLI tool for GitHub issue tracking"
  "Build a plugin-based architecture for data exporters"
  "Develop a testing framework for backend APIs"
  "Create a dynamic form generator from JSON schema"
  "Build a serverless function to resize images"
  "Create a user permission and roles service"
  "Develop a crypto price tracker and alert system"
  "Implement a notification system with fallback SMS"
  "Build a marketplace for freelance developers"
  "Create a RESTful API for managing Kubernetes clusters"
  "Develop a browser extension for blocking ads"
  "Build a chatbot that summarizes articles"
  "Create an analytics service for clickstream data"
)

# Loop through and send POST request for each description
for i in "${!DESCRIPTIONS[@]}"; do
  echo "🔁 Request $((i+1)):"
  curl -s -X POST "$URL" \
       -H "Content-Type: application/json" \
       -d "{\"description\": \"${DESCRIPTIONS[$i]}\"}" \
       | jq '.'
  echo "-----------------------------"
done
