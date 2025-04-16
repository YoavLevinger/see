#!/bin/bash

# Endpoint
URL="http://localhost:8080/process"

# List of 32 new software task descriptions
DESCRIPTIONS=(
  "Build a RESTful API for an online book library"
  "Implement an authentication service with JWT tokens"
  "Create a markdown editor with real-time preview"
  "Develop a microservice for video transcoding"
  "Build a cron-based email newsletter scheduler"
  "Create a P2P file-sharing system in Python"
  "Develop a Dockerized machine learning model server"
  "Build a backend service to manage invoices and billing"
  "Implement a caching layer using Redis for API responses"
  "Create a tool to convert CSV to Excel and vice versa"
  "Develop a chatbot that helps with DevOps questions"
  "Build a REST API for a movie recommendation engine"
  "Implement a multi-tenant SaaS backend with Flask"
  "Create an SMS verification system for signups"
  "Develop a CLI for managing project scaffolding"
  "Build a URL shortener with analytics tracking"
  "Implement an event-driven microservice with Kafka"
  "Create a real-time multiplayer game backend"
  "Develop a license key validation system for software"
  "Build an endpoint that classifies text sentiment"
  "Create a service that monitors website uptime"
  "Develop a resume parser using NLP"
  "Build a REST API to support booking and availability"
  "Implement a web scraper for product price tracking"
  "Create a report generator for Git repository statistics"
  "Develop a billing calculator for cloud usage"
  "Build a geocoding service wrapper using OpenStreetMap"
  "Implement a backend for personal finance tracking"
  "Create an S3-compatible file upload microservice"
  "Develop a permissions matrix builder for enterprise users"
  "Build a PDF to speech converter API"
  "Create a webhook relay and monitor tool"
)

# Send the POST request for each new description
for i in "${!DESCRIPTIONS[@]}"; do
  echo "🔁 Request $((i+1)):"
  echo "Description = [ ${DESCRIPTIONS[$i]} ]"
  curl -s -X POST "$URL" \
       -H "Content-Type: application/json" \
       -d "{\"description\": \"${DESCRIPTIONS[$i]}\"}" \
       | jq '.'
  echo "-----------------------------"
done
