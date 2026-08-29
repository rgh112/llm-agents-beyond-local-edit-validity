from benchmark_v4.models.base_model import BaseModel, ModelResponse, Message

__all__ = ["BaseModel", "ModelResponse", "Message", "APIModel"]


def __getattr__(name):
    if name == "APIModel":
        from benchmark_v4.models.api_model import APIModel
        return APIModel
    raise AttributeError(name)
