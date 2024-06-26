import enum
import os

"""
If we are running in a mode where each request should provide their own credentials
"""
SAAS_MODE = os.environ.get("OPTIMODEL_SAAS_MODE", None)
