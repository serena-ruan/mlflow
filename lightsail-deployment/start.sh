#!/bin/bash
set -e

# Start MLflow server in background (uses uvicorn by default)
echo "Starting MLflow server..."
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 0.0.0.0 \
  --port 5000 \
  --disable-security-middleware &

# Wait for MLflow to be ready
echo "Waiting for MLflow server to start..."
for i in {1..30}; do
  if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "MLflow server is ready!"
    break
  fi
  sleep 1
done

echo "MLflow server started successfully!"

# Generate demo data BEFORE starting nginx (ensures data is ready)
echo "Generating demo data..."
curl -s -X POST http://localhost:5000/ajax-api/3.0/mlflow/demo/generate > /tmp/demo-generation.log 2>&1
if [ $? -eq 0 ]; then
  echo "Demo data generated successfully!"
  cat /tmp/demo-generation.log
else
  echo "Demo data generation failed (continuing anyway):"
  cat /tmp/demo-generation.log
fi

# Test nginx configuration
echo "Testing nginx configuration..."
nginx -t

# Start nginx in foreground (main process - must stay running)
echo "Starting nginx on port 80..."
exec nginx -g 'daemon off;'
