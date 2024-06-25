import uvicorn
from .index import app


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)
