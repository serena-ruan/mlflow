#!/bin/bash
set -e

# MLflow Demo - Access Control Testing Script
# This script tests that the allowlist-based security is working correctly

# Configuration
SERVICE_NAME="mlflow-demo"
REGION="us-east-1"

echo "========================================="
echo "MLflow Demo - Access Control Testing"
echo "========================================="
echo ""

# Get the public URL
echo "Getting MLflow demo URL..."
MLFLOW_URL=$(aws lightsail get-container-services \
    --service-name $SERVICE_NAME \
    --query 'containerServices[0].url' \
    --output text \
    --region $REGION 2>/dev/null)

if [ -z "$MLFLOW_URL" ]; then
    echo "Error: Could not get service URL. Is the service deployed?"
    echo ""
    echo "Try running:"
    echo "  aws lightsail get-container-services --service-name $SERVICE_NAME --region $REGION"
    exit 1
fi

# Remove trailing slash if present
MLFLOW_URL=${MLFLOW_URL%/}

echo "Testing: $MLFLOW_URL"
echo ""

# Test counter
PASSED=0
FAILED=0

# Helper function to run tests
run_test() {
    local test_name="$1"
    local expected_code="$2"
    local curl_cmd="$3"

    echo "Test: $test_name"
    echo "  Command: $curl_cmd"

    # Run curl and capture HTTP code
    HTTP_CODE=$(eval "$curl_cmd" 2>/dev/null)

    if [ "$HTTP_CODE" = "$expected_code" ]; then
        echo "  ✓ PASS: Got expected $expected_code"
        ((PASSED++))
    else
        echo "  ✗ FAIL: Expected $expected_code, got $HTTP_CODE"
        ((FAILED++))
    fi
    echo ""
}

echo "========================================="
echo "Testing Allowed Operations"
echo "========================================="
echo ""

# Test 1: UI should load
run_test \
    "UI loads (GET /)" \
    "200" \
    "curl -s -o /dev/null -w '%{http_code}' '$MLFLOW_URL/'"

# Test 2: Health check
run_test \
    "Health check (GET /health)" \
    "200" \
    "curl -s -o /dev/null -w '%{http_code}' '$MLFLOW_URL/health'"

# Test 3: Search traces (required endpoint)
run_test \
    "Search traces (POST /api/2.0/mlflow/traces/search-traces)" \
    "200" \
    "curl -s -o /dev/null -w '%{http_code}' -X POST '$MLFLOW_URL/api/2.0/mlflow/traces/search-traces' -H 'Content-Type: application/json' -d '{}'"

# Test 4: Search experiments
run_test \
    "Search experiments (POST /api/2.0/mlflow/experiments/search)" \
    "200" \
    "curl -s -o /dev/null -w '%{http_code}' -X POST '$MLFLOW_URL/api/2.0/mlflow/experiments/search' -H 'Content-Type: application/json' -d '{}'"

# Test 5: Search runs
run_test \
    "Search runs (POST /api/2.0/mlflow/runs/search)" \
    "200" \
    "curl -s -o /dev/null -w '%{http_code}' -X POST '$MLFLOW_URL/api/2.0/mlflow/runs/search' -H 'Content-Type: application/json' -d '{}'"

# Test 6: List experiments
run_test \
    "List experiments (GET /api/2.0/mlflow/experiments/list)" \
    "200" \
    "curl -s -o /dev/null -w '%{http_code}' '$MLFLOW_URL/api/2.0/mlflow/experiments/list'"

echo "========================================="
echo "Testing Blocked Operations"
echo "========================================="
echo ""

# Test 7: Create experiment should be blocked
run_test \
    "Create experiment (POST /api/2.0/mlflow/experiments/create) - should be BLOCKED" \
    "403" \
    "curl -s -o /dev/null -w '%{http_code}' -X POST '$MLFLOW_URL/api/2.0/mlflow/experiments/create' -H 'Content-Type: application/json' -d '{\"name\":\"test\"}'"

# Test 8: Create run should be blocked
run_test \
    "Create run (POST /api/2.0/mlflow/runs/create) - should be BLOCKED" \
    "403" \
    "curl -s -o /dev/null -w '%{http_code}' -X POST '$MLFLOW_URL/api/2.0/mlflow/runs/create' -H 'Content-Type: application/json' -d '{\"experiment_id\":\"0\"}'"

# Test 9: Log metric should be blocked
run_test \
    "Log metric (POST /api/2.0/mlflow/runs/log-metric) - should be BLOCKED" \
    "403" \
    "curl -s -o /dev/null -w '%{http_code}' -X POST '$MLFLOW_URL/api/2.0/mlflow/runs/log-metric' -H 'Content-Type: application/json' -d '{\"run_id\":\"123\",\"key\":\"test\",\"value\":1.0,\"timestamp\":0}'"

# Test 10: Update run should be blocked
run_test \
    "Update run (POST /api/2.0/mlflow/runs/update) - should be BLOCKED" \
    "403" \
    "curl -s -o /dev/null -w '%{http_code}' -X POST '$MLFLOW_URL/api/2.0/mlflow/runs/update' -H 'Content-Type: application/json' -d '{\"run_id\":\"123\",\"status\":\"FINISHED\"}'"

# Test 11: Delete experiment should be blocked
run_test \
    "Delete experiment (POST /api/2.0/mlflow/experiments/delete) - should be BLOCKED" \
    "403" \
    "curl -s -o /dev/null -w '%{http_code}' -X POST '$MLFLOW_URL/api/2.0/mlflow/experiments/delete' -H 'Content-Type: application/json' -d '{\"experiment_id\":\"0\"}'"

# Test 12: Non-allowlisted endpoint should be blocked
run_test \
    "Unknown endpoint (POST /api/2.0/mlflow/unknown-endpoint) - should be BLOCKED" \
    "403" \
    "curl -s -o /dev/null -w '%{http_code}' -X POST '$MLFLOW_URL/api/2.0/mlflow/unknown-endpoint' -H 'Content-Type: application/json' -d '{}'"

# Test 13: PUT method should be blocked
run_test \
    "PUT method (PUT /api/2.0/mlflow/experiments/list) - should be BLOCKED" \
    "403" \
    "curl -s -o /dev/null -w '%{http_code}' -X PUT '$MLFLOW_URL/api/2.0/mlflow/experiments/list'"

# Test 14: DELETE method should be blocked
run_test \
    "DELETE method (DELETE /api/2.0/mlflow/experiments/list) - should be BLOCKED" \
    "403" \
    "curl -s -o /dev/null -w '%{http_code}' -X DELETE '$MLFLOW_URL/api/2.0/mlflow/experiments/list'"

echo "========================================="
echo "Test Summary"
echo "========================================="
echo ""
echo "Total tests: $((PASSED + FAILED))"
echo "✓ Passed: $PASSED"
echo "✗ Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 All tests passed! Access controls are working correctly."
    echo ""
    echo "Your MLflow demo is available at:"
    echo "  $MLFLOW_URL"
    echo ""
    exit 0
else
    echo "⚠️  Some tests failed. Please review the configuration."
    echo ""
    echo "Check nginx logs:"
    echo "  aws lightsail get-container-log \\"
    echo "    --service-name $SERVICE_NAME \\"
    echo "    --container-name mlflow \\"
    echo "    --region $REGION"
    echo ""
    exit 1
fi
