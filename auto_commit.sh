#!/bin/bash


# Default commit message (timestamp + short message)
msg="Auto commit: $(date '+%Y-%m-%d %H:%M:%S') - updated code & results"

git add . &&
git commit -m "$msg" &&
git push origin
