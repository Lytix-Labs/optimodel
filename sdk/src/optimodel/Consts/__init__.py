import os


BASE_URL = os.environ.get(
    "OPTIMODEL_BASE_URL", "https://api.lytix.co/optimodel/api/v1/"
)
LX_API_KEY = os.environ.get("LX_API_KEY", None)
