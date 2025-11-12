#!/bin/bash


# Default commit message (timestamp + short message)
msg="Auto commit: $(date '+%Y-%m-%d %H:%M:%S') - updated code & results" 

# Allow optional argument as custom commit message
if [ "$#" -ge 1 ]; then
  msg="$*"
fi

PATH_TO_REPO="/local/scratch/zyu273/alkd" 

# Add all changed files (code + results)
git add $PATH_TO_REPO/* &

# Commit changes
git commit -m "$msg" &

# Push to the current branch
branch=$(git rev-parse --abbrev-ref HEAD)
git push origin "$branch" &

echo "✅ Commit pushed to $branch: $msg"
