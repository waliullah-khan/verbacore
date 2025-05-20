#!/bin/bash

# This script helps to run the dashboard locally
# Make sure to run 'chmod +x run_dashboard.sh' before executing

set -e # Exit on error

# Check if Python is installed
if ! [ -x "$(command -v python3)" ] && ! [ -x "$(command -v python)" ]; then
  echo 'Error: python is not installed.' >&2
  exit 1
fi

# Determine which Python command to use
if [ -x "$(command -v python3)" ]; then
  PY_CMD="python3"
else
  PY_CMD="python"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  $PY_CMD -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Run the dashboard
echo "Starting Streamlit dashboard..."
streamlit run dashboard_app.py

# Deactivate virtual environment (this line won't be reached until the dashboard is closed)
deactivate