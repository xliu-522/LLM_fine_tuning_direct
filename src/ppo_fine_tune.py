import os
import pandas as pd
import torch
import time
from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import Dataset, DataLoader
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
from trl.core import respond_to_batch, LengthSampler
import transformers
import peft
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig


class PPO_Fine_Tune(object):
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
        self.kl_penalty = config['PPO_train']['kl_penalty']
        self.learning_rate = config['PPO_train']['learning_rate']
        self.lora_r = config['lora']['lora_r']
        self.lora_alpha = config['lora']['lora_alpha']
        self.lora_dropout = config['lora']['lora_dropout']
        self.metric_translation = {int(key): value for key, value in config['model']['metric_translation'].items()}
        
    
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
                lora_dropout=self.lora_dropout,
                bias="none",
                target_modules=["q", "v"],
                task_type=TaskType.SEQ_2_SEQ_LM
            )
        else:
            quantization_config = None
            peft_config = None

        model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                peft_config=peft_config,
        )


        model_ref = create_reference_model(model)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        #initialize trainer
        ppo_config = PPOConfig(
            model_name=self.model_name, 
            log_with="wandb", 
            batch_size=1,
            kl_penalty=self.kl_penalty,
            learning_rate=self.learning_rate)
        
        ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)
        
        for i in range(1):
            for index, row in tqdm(self.train.iterrows()):
                time.sleep(0)
                #encode query
                query_tensor = tokenizer(row["prompt"], return_tensors="pt").to(device)

                #get response form LM
                response_tensor = model.generate(**query_tensor, max_new_tokens=10)

                #pad the response
                padded_response_tensor = torch.hstack(
                    [response_tensor,
                     torch.tensor([[tokenizer.pad_token_id] * 3]).to('cuda')])

                response = tokenizer.batch_decode(padded_response_tensor, skip_special_tokens=True)[0]

                response = self.tag_translation.get(response)
                if response in self.tag_translation.values():
                    response = self.metric_translation[response]
                    labels = self.metric_translation[row['labels']]

                    #define a reward for response

                    if response == labels:
                        reward = [torch.tensor(1.0)]
                    else:
                        reward = [torch.tensor(-1.0)]

                    # train model for one step with ppo
                    train_stats = ppo_trainer.step([query_tensor['input_ids'][0]], [padded_response_tensor[0]], reward)
                    ppo_trainer.log_stats(train_stats, row, reward)
                    torch.cuda.empty_cache()
                    #print("**** finish one ppo training ****")
                else:
                    print('response not found')
        if self.quantization:
            output_dir = os.getcwd()+"/models/"+self.base_model_name+"_"+self.fine_tuning_algo+"_fine_tune_quant"+"_"+str(self.quantization_bit)
        else:
            output_dir=os.getcwd()+"/models/"+self.base_model_name+"_"+self.fine_tuning_algo+"_fine_tune_noquant"
        
        #ppo_trainer.save_pretrained(output_dir)
        os.makedirs(output_dir, exist_ok = True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)