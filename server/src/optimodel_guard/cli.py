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
    parser.add_argument(
        "--init", action="store_true", help="Initialize the model guards"
    )
    args = parser.parse_args()

    if args.config:
        os.environ["OPTIMODEL_CONFIG_PATH"] = args.config
    if args.init:
        """
        Import our guards to ensure they are initialized
        """
        from optimodel_guard.Guards import GuardMapping
        from huggingface_hub import login

        HF_TOKEN = os.environ.get("HF_TOKEN")
        if HF_TOKEN:
            login(HF_TOKEN)

        print("Done init model guards ✅")
        return

    uvicorn.run(
        "optimodel_guard.index:app", host="0.0.0.0", port=8001, workers=args.workers
    )
