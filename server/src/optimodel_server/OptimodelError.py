class OptimodelError(Exception):
    """
    Custom error class for Optimodel server errors.
    """

    def __init__(self, message: str, provider: str = None):
        super().__init__(message)
        self.provider = provider
        self.message = message

    def __str__(self):
        if self.provider:
            return f"OptimodelError with provider {self.provider}: {self.message}"
        return f"OptimodelError: {self.message}"
