#!/bin/bash

# This script helps to initialize a git repository and push it to GitHub
# Make sure to run 'chmod +x setup_github.sh' before executing

set -e # Exit on error

# Check if git is installed
if ! [ -x "$(command -v git)" ]; then
  echo 'Error: git is not installed.' >&2
  exit 1
fi

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
  echo "Initializing git repository..."
  git init
else
  echo "Git repository already initialized."
fi

# Add all files to git
echo "Adding files to git..."
git add .

# Make initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: Patronus Dashboard"

# Prompt for GitHub repository details
read -p "Enter your GitHub username: " username
read -p "Enter the name of your GitHub repository: " repo_name

# Check if the remote already exists
if git remote | grep -q "origin"; then
  echo "Remote 'origin' already exists. Updating URL..."
  git remote set-url origin "https://github.com/$username/$repo_name.git"
else
  echo "Adding remote 'origin'..."
  git remote add origin "https://github.com/$username/$repo_name.git"
fi

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main || git push -u origin master

echo "Setup complete! Your code is now on GitHub at https://github.com/$username/$repo_name"
echo "To deploy to Vercel, follow the instructions in the README.md file."