import json

# from optimodel_types import GuardQueryBase


class GuardObjectEncoder(json.JSONEncoder):
    def default(self, o):
        # if isinstance(o, GuardQueryBase):
        #     return o.__dict__
        return o
