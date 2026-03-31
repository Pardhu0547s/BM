#!/bin/bash
# Train the model and launch the dashboard in one step

echo "🚀 Starting OsteoScan Pipeline..."

# 1. Train the model (use --fast if you want to skip full grid search)
echo "🧠 Training model..."
python -m src.train --fast

# 2. Check if training succeeded
if [ $? -eq 0 ]; then
    echo "✅ Training complete! Launching FastAPI & Vite dashboard..."
    
    # Launch API Backend
    uvicorn server:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    
    # Launch React Frontend
    cd frontend && npm run dev &
    FRONTEND_PID=$!
    
    # Keep script running
    wait $BACKEND_PID
    wait $FRONTEND_PID
else
    echo "❌ Training failed. Aborting dashboard launch."
    exit 1
fi
