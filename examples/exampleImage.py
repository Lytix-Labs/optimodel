from optimodel import queryModel, listModels
from optimodel_server_types import (
    ModelImageMessageSource,
    ModelMessage,
    ModelMessageContentEntry,
    ModelTypes,
    Providers,
)

import json
import asyncio
import logging
import base64
import os


logging.basicConfig(level=logging.INFO)


current_file_path = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_file_path, "image-test.png")

with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())
    encoded_string = encoded_string.decode("utf-8")


async def main():
    prompt = "Describe the image in a JSON format"

    response = await queryModel(
        model=ModelTypes.claude_3_5_sonnet,
        messages=[
            ModelMessage(
                role="system",
                content="You are a helpful assistant. Always respond in JSON syntax",
            ),
            ModelMessage(
                role="user",
                content=[
                    ModelMessageContentEntry(type="text", text=prompt),
                    ModelMessageContentEntry(
                        type="image",
                        source=ModelImageMessageSource(
                            type="base64",
                            mediaType="image/png",
                            data=encoded_string,
                        ),
                    ),
                ],
            ),
        ],
        speedPriority="high",
        maxGenLen=1024,
        provider=Providers.bedrock,
    )

    print("Got response:", response)


async def listModelsMain():
    models = await listModels()
    print(f"All models: {json.dumps(models, indent=4)}")


if __name__ == "__main__":
    asyncio.run(main())
