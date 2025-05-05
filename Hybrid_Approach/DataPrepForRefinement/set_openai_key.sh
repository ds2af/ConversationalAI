#!/bin/bash

# Read the API key from the file
API_KEY=$(<OpenAIAPIKEY.txt)

# Export the API key as an environment variable
export OPENAI_API_KEY="$API_KEY"

# Optional: Print a confirmation (don't print the key itself for security)
echo "OPENAI_API_KEY has been set from OpenAIAPIKEY.txt"

