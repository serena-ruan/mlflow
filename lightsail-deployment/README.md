# Deploy Read-Only MLflow Demo on AWS Lightsail

This guide shows how to deploy a read-only MLflow demo server on AWS Lightsail with the following characteristics:

## Requirements Met

✅ **1. All users can access the UI** - Public HTTPS endpoint provided by Lightsail
✅ **2. The UI is read-only** - nginx blocks all write operations (create, update, delete, log)
✅ **3. Search traces operation allowed** - POST to search endpoints explicitly allowlisted

## Architecture

- **Compute**: AWS Lightsail Container Service (medium tier for testing, $40/month - can downgrade later)
- **Storage**: Ephemeral SQLite in container (demo data resets on restart)
- **Demo Data**: Pre-populated using `mlflow demo` command
- **Access Control**: nginx reverse proxy with **allowlist-based security model**
  - Only explicitly allowed endpoints are accessible
  - New MLflow APIs are blocked by default until allowlisted
  - Prevents unintended access when MLflow adds new features
- **HTTPS**: Automatic SSL certificate included with Lightsail

## What's Allowed vs Blocked

This configuration uses an **allowlist-based security model**: only explicitly allowed endpoints work, everything else is blocked by default. This ensures that new MLflow APIs added in the future will be automatically blocked until explicitly allowlisted.

### ✅ Allowed Operations (Explicit Allowlist)

**Search Operations (POST allowed to specific endpoints)**:
- `POST /api/2.0/mlflow/traces/search-traces` - Search traces ✅ **Required**
- `POST /api/2.0/mlflow/experiments/search` - Search experiments (needed for UI)
- `POST /api/2.0/mlflow/runs/search` - Search runs (needed for UI)

**Read Operations (GET only)**:
- `/api/2.0/mlflow/experiments/get` - Get experiment by ID
- `/api/2.0/mlflow/experiments/get-by-name` - Get experiment by name
- `/api/2.0/mlflow/experiments/list` - List experiments
- `/api/2.0/mlflow/runs/get` - Get run by ID
- `/api/2.0/mlflow/runs/list` - List runs
- `/api/2.0/mlflow/metrics/*` - Get metrics
- `/api/2.0/mlflow/artifacts/*` - Get artifacts
- `/api/2.0/mlflow/model-versions/get` - Get model version
- `/api/2.0/mlflow/model-versions/search` - Search model versions
- `/api/2.0/mlflow/model-versions/get-download-uri` - Get model download URI
- `/api/2.0/mlflow/registered-models/get` - Get registered model
- `/api/2.0/mlflow/registered-models/search` - Search registered models
- `/api/2.0/mlflow/registered-models/list` - List registered models
- `/api/2.0/mlflow/registered-models/get-latest-versions` - Get latest model versions
- `/api/2.0/mlflow/traces/get` - Get trace by ID
- `/api/2.0/mlflow/traces/get-trace-artifact` - Get trace artifact
- `/health` - Health check
- `/` - UI static assets

### ❌ Blocked Operations (Everything Else)

**All write operations** (explicitly blocked):
- Create, update, delete experiments
- Create, update, delete runs
- Log metrics, parameters, tags, models
- Create, update, delete registered models
- Create, update, delete model versions
- Any POST/PUT/DELETE/PATCH to non-allowlisted endpoints

**All new APIs** (blocked by default):
- Any future MLflow API endpoints will be blocked until explicitly added to the allowlist above
- This is a security feature to prevent unintended access to new functionality

## Deployment Files

**All deployment files are available in the `lightsail-deployment/` directory:**

```
lightsail-deployment/
├── Dockerfile                   # Container image with MLflow + nginx
├── nginx.conf                   # Allowlist-based reverse proxy config
├── start.sh                     # Startup script (MLflow + nginx)
├── deployment.json              # Lightsail deployment configuration
└── test-access-controls.sh      # Automated testing script
```

All configuration files are provided in the directory - see the files for implementation details.

## Manual Deployment Steps

If you prefer to deploy manually instead of using the automated script:

### Prerequisites

- AWS CLI installed and configured
- Docker installed locally
- AWS Lightsail access

### Step 1: Navigate to Deployment Directory

```bash
cd lightsail-deployment/
```

### Step 2: Create Lightsail Container Service

```bash
aws lightsail create-container-service \
  --service-name mlflow-demo \
  --power medium \
  --scale 1 \
  --region us-east-1
```

**Note**: The medium tier ($40/month) includes:
- 2 vCPU
- 2 GB RAM
- 500 GB data transfer

*Using medium tier for testing to ensure demo data generation works smoothly. Can downgrade to small ($10/month, 1GB) later if it works.*

### Step 3: Build and Push Docker Image

```bash
# Build the image for linux/amd64 (required by Lightsail)
docker build --no-cache --platform linux/amd64 -t mlflow-demo .

# Push to Lightsail
aws lightsail push-container-image \
  --service-name mlflow-demo \
  --label mlflow-latest \
  --image mlflow-demo:latest \
  --region us-east-1
```

**Important**: Note the image reference from the output. It will look like:
```
:mlflow-demo.mlflow-latest.X
```

### Step 4: Update deployment.json

The `deployment.json` file is already provided in the directory. Just update the image reference with the value from Step 3:

```json
{
  "containers": {
    "mlflow": {
      "image": ":mlflow-demo.mlflow-latest.1",  // <-- Update this
      "ports": {
        "80": "HTTP"
      }
    }
  },
  "publicEndpoint": {
    "containerName": "mlflow",
    "containerPort": 80,
    "healthCheck": {
      "healthyThreshold": 2,
      "unhealthyThreshold": 2,
      "timeoutSeconds": 5,
      "intervalSeconds": 30,
      "path": "/health",
      "successCodes": "200-499"
    }
  }
}
```

### Step 5: Deploy the Container

```bash
aws lightsail create-container-service-deployment \
  --service-name mlflow-demo \
  --cli-input-json file://deployment.json \
  --region us-east-1
```

### Step 6: Wait for Deployment

Check deployment status:

```bash
aws lightsail get-container-services \
  --service-name mlflow-demo \
  --region us-east-1
```

Wait until `state` shows `RUNNING` (typically 5-10 minutes).

### Step 7: Get Public URL

```bash
aws lightsail get-container-services \
  --service-name mlflow-demo \
  --query 'containerServices[0].url' \
  --output text \
  --region us-east-1
```

Your MLflow demo will be available at:
```
https://mlflow-demo.xxxxx.us-east-1.cs.amazonlightsail.com
```

## Testing Access Controls

### Test 1: UI Access (Should Work)
```bash
curl -I https://your-lightsail-url.com
# Expected: 200 OK
```

### Test 2: Search Traces (Should Work)
```bash
curl -X POST https://your-lightsail-url.com/api/2.0/mlflow/traces/search-traces \
  -H "Content-Type: application/json" \
  -d '{}'
# Expected: 200 OK with results
```

### Test 3: Create Experiment (Should Be Blocked)
```bash
curl -X POST https://your-lightsail-url.com/api/2.0/mlflow/experiments/create \
  -H "Content-Type: application/json" \
  -d '{"name": "test"}'
# Expected: 403 Forbidden
```

### Test 4: Log Metric (Should Be Blocked)
```bash
curl -X POST https://your-lightsail-url.com/api/2.0/mlflow/runs/log-metric \
  -H "Content-Type: application/json" \
  -d '{"run_id": "123", "key": "test", "value": 1.0}'
# Expected: 403 Forbidden
```

## Optional: Custom Domain

To use a custom domain:

1. In Lightsail console, go to your container service
2. Click "Custom domains" tab
3. Click "Create certificate"
4. Add your domain (e.g., `demo.mlflow.org`)
5. Follow DNS validation steps
6. Once validated, attach the certificate to your container service

## Monitoring and Logs

### View Container Logs

```bash
aws lightsail get-container-log \
  --service-name mlflow-demo \
  --container-name mlflow \
  --region us-east-1
```

### View Service Metrics

In the Lightsail console, navigate to:
- Container service → Metrics tab
- View CPU utilization, memory, and request counts

## Cost Breakdown

| Item | Cost |
|------|------|
| Lightsail Container (medium for testing) | $40.00/month |
| Data transfer (500GB included) | $0.00 |
| HTTPS certificate | $0.00 |
| Load balancer | $0.00 (included) |
| **Total (testing)** | **$40.00/month** |
| **Total (if downgraded to small)** | **$20.00/month** |

## Limitations

1. **Ephemeral storage**: Data resets on container restarts
   - Demo data is regenerated on each restart via API
   - For persistent data, add Lightsail managed database (+$15/month, total $25/month)

2. **No custom authentication**: All users have read access
   - nginx provides endpoint-level restrictions only
   - For user authentication, consider MLflow's built-in auth or add AWS WAF

3. **Single container**: No high availability
   - Can scale to multiple containers if needed (still on Lightsail)

## Updating the Demo

### Adding New Endpoints to the Allowlist

When MLflow adds new APIs or you want to allow additional endpoints:

1. Edit `nginx.conf` and add the endpoint to the appropriate section:

```nginx
# Example: Allow a new read-only endpoint
location = /api/2.0/mlflow/new-endpoint/get-data {
    limit_except GET HEAD OPTIONS {
        deny all;
    }
    proxy_pass http://localhost:5000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

# Example: Allow a new POST search endpoint
location = /api/2.0/mlflow/new-search {
    limit_except GET HEAD OPTIONS POST {
        deny all;
    }
    proxy_pass http://localhost:5000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

2. Rebuild and redeploy (see below)

### Updating Container Configuration

To update the container with new code/config:

```bash
# Rebuild and push
docker build --platform linux/amd64 -t mlflow-demo .
aws lightsail push-container-image \
  --service-name mlflow-demo \
  --label mlflow-latest \
  --image mlflow-demo:latest \
  --region us-east-1

# Redeploy (update image reference in deployment.json)
aws lightsail create-container-service-deployment \
  --service-name mlflow-demo \
  --cli-input-json file://deployment.json \
  --region us-east-1
```

## Cleanup

To delete the service and stop charges:

```bash
aws lightsail delete-container-service \
  --service-name mlflow-demo \
  --region us-east-1
```

## Troubleshooting

### Container won't start
- Check logs: `aws lightsail get-container-log --service-name mlflow-demo --container-name mlflow`
- Verify image was pushed successfully
- Ensure port 80 is exposed in Dockerfile

### 403 errors on legitimate requests
- Check nginx.conf regex patterns
- Verify endpoint paths match MLflow API exactly
- Test locally with Docker first

### Health check failures
- MLflow `/health` endpoint may take time to respond
- Increase `timeoutSeconds` and `intervalSeconds` in health check
- Or change path to `/` which always responds

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [AWS Lightsail Container Service](https://aws.amazon.com/lightsail/features/)
- [MLflow Server Configuration](https://mlflow.org/docs/latest/tracking/server.html)
