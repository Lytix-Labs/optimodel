from optimodel_types import ModelMessage, ModelMessageContentEntry


def isStringOrTextType(message: ModelMessage):
    return isinstance(message.content, str) or allMessageContentIsText(message.content)


def allMessageContentIsText(messages: list[ModelMessageContentEntry]):
    return all(x.type == "text" for x in messages)


def containsImageInMessages(messages: list[ModelMessage]):
    return any(not isStringOrTextType(x) for x in messages)
