from mlml_hugginface import Downloader
from mlml_hugginface.train import Trainer
import sys
import os
from dataclasses import dataclass
import json


@dataclass
class TrainerConfig:
    MODEL_CHECKPOINT: str
    DATASET_NAME: str
    HF_CHECKPOINT: str = None
    LORA: bool = False
    MLM_PROBABILITY: float = 0.15
    BATCH_SIZE: int = 16

    def __post_init__(self):
        for field_key in self.__dataclass_fields__.keys():
            if self.__getattribute__(field_key) is None:
             raise ValueError(f'missing {field_key} config property')


training_config_jsonStr_or_fp = "".join(sys.argv[1:])

if os.path.exists(training_config_jsonStr_or_fp):
    with open(training_config_jsonStr_or_fp) as inpf:
        config = json.load(inpf)
else:
        config = json.loads(training_config_jsonStr_or_fp)
    
config = trainerConfig(**{k.upper(): v for k, v in config.items()})
print(trainerConfig);input("start training pipe?")
trainer = Trainer(training_config_fp)
trainer.train()
