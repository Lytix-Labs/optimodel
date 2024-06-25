import enum
import os


class ModelTypes(enum.Enum):
    llama3_8b_instruct = "llama3_8b_instruct"
    llama3_8b_chat = "llama3_8b_chat"
    llama3_70b_instruct = "llama3_70b_instruct"


"""
If we are running in a mode where each request should provide their own credentials
"""
SAAS_MODE = os.environ.get("OPTIMODEL_SAAS_MODE", None)
