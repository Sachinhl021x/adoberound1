#!/bin/bash
# Startup script for AI Leadership Insight Agent
# Starts both FastAPI backend and Streamlit frontend (enhanced UI)

set -e

echo "🚀 Starting AI Leadership Insight Agent..."
echo "=========================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "   Please edit .env and add your OPENAI_API_KEY"
    read -p "   Press Enter after setting up .env..."
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv312" ]; then
        echo "✓ Activating .venv312..."
        source .venv312/bin/activate
    else
        echo "⚠️  Warning: No virtual environment detected."
        echo "   Consider activating venv: source .venv312/bin/activate"
        read -p "   Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

# Check if index exists
if [ ! -d "indexes/chromadb" ]; then
    echo "❌ Error: Vector index not found at indexes/chromadb"
    echo "   Please run: python scripts/build_index.py"
    exit 1
fi

# Start FastAPI in background
echo ""
echo "[1/2] Starting FastAPI backend..."
python scripts/serve.py &
FASTAPI_PID=$!

# Wait for FastAPI to be ready
echo "   Waiting for API to be ready..."
sleep 3

# Check if FastAPI is running
if ! ps -p $FASTAPI_PID > /dev/null; then
    echo "❌ FastAPI failed to start"
    exit 1
fi

# Start Streamlit
echo ""
echo "[2/2] Starting Streamlit frontend..."
echo ""
streamlit run app.py

# Cleanup: Kill FastAPI when Streamlit exits
echo ""
echo "🛑 Shutting down..."
kill $FASTAPI_PID 2>/dev/null || true
echo "✓ Services stopped"
