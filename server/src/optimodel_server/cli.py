import uvicorn
from pathlib import Path
import os


import argparse


def main():
    parser = argparse.ArgumentParser(description="Run the Uvicorn server.")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    parser.add_argument(
        "--config", type=Path, help="Path to a custom JSON configuration file"
    )
    args = parser.parse_args()

    if args.config:
        os.environ["OPTIMODEL_CONFIG_PATH"] = args.config

    uvicorn.run(
        "optimodel_server.index:app", host="0.0.0.0", port=8000, workers=args.workers
    )
