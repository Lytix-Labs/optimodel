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


class OptimodelGuardError(Exception):
    """
    Custom error class for Optimodel guard errors.
    """

    def __init__(self, message: str, guard: str = None):
        super().__init__(message)
        self.guard = guard
        self.message = message

    def __str__(self):
        if self.guard:
            return f"OptimodelGuardError with guard {self.guard}: {self.message}"
        return f"OptimodelGuardError: {self.message}"
