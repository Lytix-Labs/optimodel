from optimodel_server_types import ModelMessage


def containsImageInMessages(messages: list[ModelMessage]):
    return any(not isinstance(x.content, str) for x in messages)
