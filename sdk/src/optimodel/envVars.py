import os


class LytixEnvVars:
    LX_BASE_URL = os.environ.get("LX_BASE_URL", "https://api.lytix.co/")
    LX_API_KEY = os.environ.get("LX_API_KEY")

    def setAPIKey(self, apiKey):
        self.LX_API_KEY = apiKey

    def validateEnvVars(self):
        if self.LX_BASE_URL is None:
            print("LX_BASE_URL envvar is not set")
        if self.LX_API_KEY is None:
            print(
                "LX_API_KEY envvar is not set. Make sure to set it via LytixCreds.setAPIKey before making requests"
            )


"""
Always validate when importing this file
@note This will not hard break, just print an error
"""
LytixCreds = LytixEnvVars()
LytixCreds.validateEnvVars()
