import json

import asyncio

from optimodel import queryModel, listModels
from optimodel_server_types import ModelMessage, ModelTypes

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
        model=ModelTypes.mistral_7b_instruct,
        messages=[
            ModelMessage(
                role="system",
                content="You are a helpful assistant. Always respond in JSON syntax",
            ),
            ModelMessage(role="user", content=prompt),
        ],
        speedPriority="high",
        validator=validator,
        # fallbackModels=[ModelTypes.llama_3_70b_instruct],
        maxGenLen=256,
    )
    print("Got response:", response)


async def listModelsMain():
    models = await listModels()
    print(f"All models: {json.dumps(models, indent=4)}")


if __name__ == "__main__":
    asyncio.run(main())
