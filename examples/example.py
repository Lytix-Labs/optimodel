import json

import asyncio

from optimodel import queryModel, ModelTypes
from optimodel_server import ModelMessage

import logging

logging.basicConfig(level=logging.INFO)


def validator(x) -> bool:
    """
    Simple validator to check if the response is JSON
    """
    try:
        json.loads(x)
        return True
    except:
        return False


async def main():
    prompt = "Hello How are you?"

    response = await queryModel(
        model=ModelTypes.llama3_8b_instruct,
        messages=[
            ModelMessage(role="user", content=prompt),
        ],
        speedPriority="low",
        validator=validator,
        fallbackModels=[ModelTypes.llama3_70b_instruct],
    )
    print("Got response:", response)


if __name__ == "__main__":
    asyncio.run(main())
