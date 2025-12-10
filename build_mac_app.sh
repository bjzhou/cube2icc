#!/bin/bash
set -e

echo "Building Cube2ICC.app..."
uv run pyinstaller --noconfirm --clean --name "Cube2ICC" --windowed gui.py

echo "Build complete. App is located in dist/Cube2ICC.app"
