"""
Health check script for fks_analyze Docker container.
"""

import os
import sys

# Simple health check - extend based on actual service requirements
if __name__ == "__main__":
    # Check if required environment variables are set
    if not os.getenv("GOOGLE_AI_API_KEY"):
        print("Health check failed: GOOGLE_AI_API_KEY not set")
        sys.exit(1)
    
    # Add more health check logic here as needed
    # For now, just exit with success if the key is set
    print("Health check passed")
    sys.exit(0)
