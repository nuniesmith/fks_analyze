# FKS Analyze Service

Repository analysis and code quality service for FKS Trading Platform.

## Features

- **File Structure Analysis**: Analyze repository structure, file counts, and sizes
- **Empty Item Detection**: Find empty files and directories
- **Broken File Detection**: Identify files with syntax errors or issues
- **Code Metrics**: Calculate lines of code and other metrics
- **REST API**: FastAPI-based API for integration with Django dashboard
- **Background Jobs**: Asynchronous analysis with job tracking

## Quick Start

### Development

```bash
# Install dependencies
pip install -r requirements.dev.txt

# Run service
uvicorn src.main:app --reload --host 0.0.0.0 --port 8008
```

### Docker

```bash
# Build image
docker build -t nuniesmith/fks_analyze:latest .

# Run container
docker run -p 8008:8008 nuniesmith/fks_analyze:latest
```

## API Endpoints

### Health Checks

- `GET /health` - Health check
- `GET /health/ready` - Readiness check

### Analysis

- `POST /api/v1/analysis/run` - Run repository analysis
- `GET /api/v1/analysis/status/{job_id}` - Get job status
- `GET /api/v1/analysis/results/{job_id}` - Get job results
- `GET /api/v1/analysis/jobs` - List all jobs
- `DELETE /api/v1/analysis/jobs/{job_id}` - Delete job

## Example Usage

### Run Analysis

```bash
curl -X POST http://localhost:8008/api/v1/analysis/run \
  -H "Content-Type: application/json" \
  -d '{
    "repository_path": "/home/jordan/Documents/code/fks",
    "include_mermaid": false,
    "include_lint": false
  }'
```

Response:
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "queued",
  "message": "Analysis job queued",
  "timestamp": "2025-11-07T12:00:00Z"
}
```

### Check Status

```bash
curl http://localhost:8008/api/v1/analysis/status/123e4567-e89b-12d3-a456-426614174000
```

### Get Results

```bash
curl http://localhost:8008/api/v1/analysis/results/123e4567-e89b-12d3-a456-426614174000
```

## Integration with Django Dashboard

See [Django Integration Guide](docs/DJANGO_INTEGRATION.md) for instructions on integrating with the FKS web dashboard.

## Configuration

Configuration via environment variables or `.env` file:

- `DEBUG` - Debug mode (default: False)
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8008)
- `CORS_ORIGINS` - Allowed CORS origins
- `MAX_FILE_SIZE_MB` - Max file size for analysis (default: 10)
- `ANALYSIS_TIMEOUT_SECONDS` - Analysis timeout (default: 300)

## Testing

```bash
pytest tests/ -v --cov=src
```

## License

See LICENSE file for details.
