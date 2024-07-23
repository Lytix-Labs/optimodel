from optimodel_server_types import ModelMessage


def containsImageInMessages(messages: list[ModelMessage]):
    return any(
        isinstance(x.content, str)
        or x.content.get("type", None) == "image"
        or x.content.get("source", None) is not None
        for x in messages
    )
