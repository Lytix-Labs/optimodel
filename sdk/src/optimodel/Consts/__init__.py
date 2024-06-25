import os


BASE_URL = os.environ.get(
    "OPTIMODEL_BASE_URL", "http://localhost:8000/optimodel/api/v1/"
)
LY_API_KEY = os.environ.get("LY_API_KEY", None)
