import os
import pandas as pd
import torch
import time
from tqdm import tqdm
import wandb
import peft
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
#from torch.utils.data import Dataset, DataLoader
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset
from trl.trainer.utils import PeftSavingCallback
import transformers


class SFT_Fine_Tune(object):
    def __init__(self, config, data, language_model_name, tag_translation):
        self.base_model_name = config['model']['base_model_name']
        self.quantization = config['model']['quantization']
        self.quantization_bit = config['model']['quantization_bit']
        if self.quantization_bit == 8:
            self.quantization_bit_4 = False
            self.quantization_bit_8 = True
        elif self.quantization_bit == 4:
            self.quantization_bit_4 = True
            self.quantization_bit_8 = False
        self.fine_tuning_algo = config['model']['fine_tuning_algo']
        self.train, self.test = train_test_split(data, test_size=1-config['data']['train_test_split_perc'], random_state=42)
        self.train.reset_index(drop=True, inplace=True)
        self.test.reset_index(drop=True, inplace=True)
        self.language_model_name = language_model_name
        self.tag_translation = tag_translation
        self.metric_translation = {int(key): value for key, value in config['model']['metric_translation'].items()}
        self.num_epoch = config['SFT_train']['num_epoch']
        self.per_device_train_batch_size=config['SFT_train']['per_device_train_batch_size'],
        self.per_device_eval_batch_size=config['SFT_train']['per_device_eval_batch_size'],
        self.evaluation_strategy=config['SFT_train']['evaluation_strategy']
        self.lora_r = config['lora']['lora_r']
        self.lora_alpha = config['lora']['lora_alpha']
        self.lora_dropout = config['lora']['lora_dropout']
        
    def train_it(self):
        print("fine tune here!")
        
        # get models
        if self.language_model_name == 'flan-t5-large':
            self.model_name = "google/flan-t5-large"
        elif self.language_model_name == 'flan-t5-xl':
            self.model_name = "google/flan-t5-xl"
        elif self.language_model_name == 'flan-t5-xxl':
            self.model_name = "google/flan-t5-xxl"
        else:
            print("warning: language_model_name not found!")
            
        if torch.cuda.is_available():
            device = 'cuda'
            device_map = 'auto'
            torch_dtype=torch.bfloat16
        else:
            devide='cpu'
            device_map = {"": torch.device("cpu")}
            torch_dtype = torch.float32
            
        if self.quantization:
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_8bit=self.quantization_bit_8, load_in_4bit=self.quantization_bit_4
            )

            device_map = {"": 0}
            torch_dtype = torch.bfloat16
            
            peft_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            
        else:
            quantization_config = None
            peft_config = None

        model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #print(tokenizer.model_max_length)
        #Prepare dataset
        ds_train = Dataset.from_pandas(self.train[["prompt", "labels"]])
        ds_train = ds_train.map(lambda samples: tokenizer(samples["prompt"]))
        ds_train.set_format(type="torch")
        
        ds_test = Dataset.from_pandas(self.test[["prompt", "labels"]])
        ds_test = ds_test.map(lambda samples: tokenizer(samples["prompt"]))
        ds_test.set_format(type="torch")
        
        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example['prompt'])):
                text = f"### Question: {example['prompt'][i]}\n ### Answer: {example['labels'][i]}"
                print(text)
                output_texts.append(text)
            return output_texts
        
#         #initialize trainers
        if self.quantization:
            output_dir = "models/"+self.base_model_name+"_"+self.fine_tuning_algo+"_fine_tune_quant"+"_"+str(self.quantization_bit)
        else:
            output_dir="models/"+self.base_model_name+"_"+self.fine_tuning_algo+"_fine_tune_noquant"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs = 0.5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_steps=10,
            eval_steps=100,
            evaluation_strategy="steps"
        )
        
            
        trainer = SFTTrainer(
            model,
            args=training_args,
            train_dataset=ds_train,
            eval_dataset=ds_test,
            dataset_text_field="prompt",
            max_seq_length=tokenizer.model_max_length,
            formatting_func=formatting_prompts_func,
            peft_config=peft_config
        )
        
        trainer.train()
        
        # model.save_pretrained(output_dir)
        # tokenizer.save_pretrained(output_dir)
        trainer.save_model(output_dir)
        