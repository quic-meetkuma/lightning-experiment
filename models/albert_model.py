# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from peft import (
    LoraConfig,
    PeftConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from models.model_factory import BaseModel, ModelRegistry


@ModelRegistry.register_model("albert")
class AlbertPEFTModel(BaseModel):
    model_name = "albert/albert-base-v2"

    def get_model(self, use_peft=False):
        if use_peft:
            try:
                peft_config = PeftConfig.from_pretrained(self.model_name)
                base_model_path = peft_config.base_model_name_or_path
                model = AutoModelForSequenceClassification.from_pretrained(
                    base_model_path
                )
            except ValueError:
                peft_config = LoraConfig(
                    task_type=TaskType.TOKEN_CLS,
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    bias="none",
                    target_modules=["query", "value"],
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=3
            )
        return model

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer
