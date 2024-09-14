from mlml_hugginface import Downloader
from mlml_hugginface.train import Trainer
import sys
import os
import json
from dataclasses import dataclass, field
from enum import Enum

# Define the enum for training strategies
class TrainingStrategy(Enum):
    FULL_LLM_TOKENIZE = "FULL+LLM-TOKENIZE"
    RESUME_LLM_TOKENIZE = "RESUME+LLM-TOKENIZE"
    FULL_HUMAN_TOKENIZE = "FULL+HUMAN-TOKENIZE"

# Reverse mapping from string to enum
# Automatically generate reverse mapping from the enum values
TRAINING_STRATEGY_MAP = {strategy.value: strategy for strategy in TrainingStrategy}

@dataclass                                                                                             
class TrainerConfig:                                                                                   
    BASE_MODEL_NAME: str = None                                                                        
    RUN_HASH: str = None                                                                               
    TRAINING_CHECKPOINT: str = None                                                                    
    DATASET_NAME: str = None                                                                           
    DATASET_FOLDER: str = "datasets"                                                                   
    SPLIT: str = None                                                                                  
    HF_CHECKPOINT: bool = False                                                                        
    LORA: bool = False                                                                                 
    MLM_PROBABILITY: float = 0.15                                                                      
    BATCH_SIZE: int = 4                                                                                
    # Allow training_strategy as a string input, which will be converted to enum                       
    TRAINING_STRATEGY: str = field(default="FULL+LLM-TOKENIZE")                                        
                                                                                                       
    def __post_init__(self):                                                                           
        required_fields = ["BASE_MODEL_NAME", "DATASET_NAME"]   for field_key in self.__dataclass_fields__.keys():
            if field_key in required_fields and self.__getattribute__(field_key) is None:
                raise ValueError(f'missing {field_key} config property')

        # Convert the string training_strategy to enum if it's a valid string
        if isinstance(self.TRAINING_STRATEGY, str):
            if self.TRAINING_STRATEGY not in TRAINING_STRATEGY_MAP:
                raise ValueError(f'Invalid training strategy: {self.TRAINING_STRATEGY}')
            self.TRAINING_STRATEGY = TRAINING_STRATEGY_MAP[self.TRAINING_STRATEGY]
        elif not isinstance(self.TRAINING_STRATEGY, TrainingStrategy):
            raise ValueError(f'Invalid training strategy type: {self.training_strategy}')

training_config_jsonStr_or_fp = "".join(sys.argv[1:])

if os.path.exists(training_config_jsonStr_or_fp):
    with open(training_config_jsonStr_or_fp) as inpf:
        config = json.load(inpf)
else:
        config = json.loads(training_config_jsonStr_or_fp)
    
config = TrainerConfig(**{k.upper(): v for k, v in config.items()})
print(TrainerConfig);input("start training pipe?")
trainer = Trainer(config)
print("*"*50,"STARTING TRAINING","*"*50)
trainer.train()
