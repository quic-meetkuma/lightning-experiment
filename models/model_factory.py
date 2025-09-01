# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from abc import ABC, abstractmethod


class ModelRegistry:
    registry = {}

    @classmethod
    def register_model(cls, model_type):
        def decorator(subclass):
            cls.registry[model_type] = subclass

        return decorator

    @staticmethod
    def fetch_model(model_type):
        if model_type in ModelRegistry.registry:
            return ModelRegistry.registry[model_type]
        else:
            raise ValueError("Unsupported model_type provided.")


class BaseModel(ABC):
    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def get_tokenizer(self):
        pass
