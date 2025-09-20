#!/usr/bin/env bash
# Build script for Render deployment

# Upgrade pip and install build tools first
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel

# Install requirements
pip install -r requirements.txt
