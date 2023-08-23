import os
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, det_curve, DetCurveDisplay, confusion_matrix
import time
import torch
from tqdm import tqdm
import peft
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig

class Evaluate(object):
    def __init__(self, config, data, language_model_name, tag_translation):
        self.config = config
        self.train, self.test = train_test_split(data, test_size=1-self.config['data']['train_test_split_perc'], random_state=42)
        self.train.reset_index(drop=True, inplace=True)
        self.test.reset_index(drop=True, inplace=True)
        self.language_model_name = language_model_name
        self.base_model_name = self.config['model']['base_model_name']
        self.fine_tuning_algo = self.config['model']['fine_tuning_algo']
        self.tag_translation = tag_translation
        self.metric_translation = {int(key): value for key, value in config['model']['metric_translation'].items()}
        self.load_fine_tune_model = config['model']['load_fine_tune_model']
        self.quantization = config['model']['quantization']
        self.quantization_bit = config['model']['quantization_bit']

    def infer(self):
        #check gpu
        if torch.cuda.is_available():
            device = 'cuda'
            device_map = 'auto'
            torch_dtype=torch.bfloat16
        else:
            devide='cpu'
            device_map = {"": torch.device("cpu")}
            torch_dtype = torch.float32
        
        #load model
        if self.load_fine_tune_model == 'fine_tune':
            if self.quantization:
                self.model_config = self.base_model_name+"_"+self.fine_tuning_algo+"_fine_tune_quant"+"_"+str(self.quantization_bit)
                model_dir = os.getcwd()+"/models/"+ self.model_config
                peft_config = PeftConfig.from_pretrained(model_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path, device_map=device_map, 
                    torch_dtype=torch_dtype)
                model = PeftModel.from_pretrained(model, model_dir)
                model = model.merge_and_unload()
                tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
            else:
                self.model_config = self.base_model_name+"_"+self.fine_tuning_algo+"_fine_tune_noquant"
                model_dir = os.getcwd()+"/models/"+self.model_config
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_dir, 
                    return_dict=True, 
                    device_map=device_map,
                    torch_dtype=torch_dtype
                )
        elif self.load_fine_tune_model == 'base_line':    
            self.model_config = self.base_model_name 
            if self.language_model_name == 'flan-t5-large':
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    "google/flan-t5-large", 
                    device_map=device_map, 
                    torch_dtype=torch_dtype
                )
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
            elif self.language_model_name == 'flan-t5-xl':
                model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl", device_map=device_map, 
                    torch_dtype=torch_dtype)
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
                #model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl", device_map="auto")
            elif self.language_model_name == 'flan-t5-xxl':
                model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl", device_map=device_map, 
                    torch_dtype=torch_dtype)
                tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
                #model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl", device_map="auto")
            else:
                print("warning: language_model_name not found!")
        else:
            print("warning: language_model_name not found!")

        inference_list = []
        parse_prediction = self.config["model"]['base_model_name'] in ["lg_name", "lg_noname", "xl_noname", "xxl_noname"]
        for index, row in tqdm(self.test.iterrows()):
            time.sleep(0)
            inputs = tokenizer(row["prompt"], return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=10)
            prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            #print(prediction)
            if parse_prediction:
                parsed_prediction = prediction
            else:
                parsed_prediction = prediction[0]
            response = self.tag_translation.get(parsed_prediction)
            if response in self.tag_translation.values():
                inference_list.append(response)
            else:
                print('response not found')
                inference_list.append('NA')
        self.test["inference"] = inference_list
        self.test.to_csv('./data/data_infer_'+self.config['model']['base_model_name']+'.csv')

    def metrics(self):
        ROC_det_confusion_metrix = self.config['model']['ROC_det_confusion_metrix']
        labels = self.test['labels'].map(self.metric_translation)
        if self.config['model']['evaluate_res'] == 'new':
            inference = self.test['inference'].map(self.metric_translation)
        elif self.config['model']['evaluate_res'] == 'old':
            inference = self.test[self.config['model']["base_model_name"]].map(self.metric_translation)
        fpr, tpr, thereshold = roc_curve(labels, inference)
        fpr_det, fnr_det, thereshold_det = det_curve(labels, inference)
        roc_auc = auc(fpr, tpr)
        time_now = time.strftime("Date_%Y_%m_%d_time_%H_%M_%S")
        lw = 2

        # ROC
        if ROC_det_confusion_metrix == "score":
            fig = plt.figure()
            plt.plot(fpr, tpr, color = "darkorange", lw=lw, label="ROC curve with AUC=%0.2f" % roc_auc)
            plt.plot([0, 1], [0, 1],color ="navy", lw=lw, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0000001])
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("ROC curve")
            plt.legend(loc="best")
            fig.savefig("./figures/"+self.config['model']['evaluate_res'] + "_"+self.config['model']['load_fine_tune_model']+"_"+ self.model_config+"_ROC_curve_"+time_now+".png", dpi=fig.dpi)
            plt.close('all')
            # det
            fig = plt.figure()
            plt.plot(fpr_det, fnr_det, color = "darkorange", lw=lw, label="ROC curve with AUC=%0.2f" % roc_auc)
            plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0000001])
            plt.xlabel("FPR")
            plt.ylabel("FNR")
            plt.title("Detection Error Tradeoff (DET) curve")
            plt.legend(loc="best")
            fig.savefig("./figures/" + self.config['model']['evaluate_res']+ "_"+self.config['model']['load_fine_tune_model']+ "_"+self.model_config + "_DET_curve_" + time_now + ".png", dpi=fig.dpi)
            plt.close('all')

        # confusion metrix
        if ROC_det_confusion_metrix == "score":
            confusion_metrix_cutoffs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999]
        elif ROC_det_confusion_metrix == "single_confusion_metrix":
            confusion_metrix_cutoffs = [0.5]

        for index, value in enumerate(confusion_metrix_cutoffs):
            y_infer_thr = (inference > value).astype(int)
            confusion_matrix_values = confusion_matrix(y_true=labels, y_pred=y_infer_thr)
            sum_first_row = confusion_matrix_values[0, :].sum()
            sum_secon_row = confusion_matrix_values[1, :].sum()
            coef_matrix_percent = np.zeros((2, 2))
            coef_matrix_percent[0, :] = confusion_matrix_values[0, :] / float(sum_first_row)
            coef_matrix_percent[1:] = confusion_matrix_values[1:] / float(sum_secon_row)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.matshow(confusion_matrix_values, cmap=plt.cm.Blues, alpha=0.3)
            for ii in range(confusion_matrix_values.shape[0]):
                for jj in range(confusion_matrix_values.shape[1]):
                    ax.text(x=jj, y=ii, s=confusion_matrix_values[ii, jj], va='center', ha='center', size='xx-large')
                    ax.text(x=jj, y=ii-0.2, s="("+"{:0.2f}".format(coef_matrix_percent[ii, jj]*100)+"%)", va='center', ha='center', size = 'xx-large')
            plt.xlabel("Predictions", fontsize=18)
            plt.ylabel("Actuals", fontsize=18)
            plt.title("Confusion matrix for "+str(value)+ " threshold", fontsize=18)
            fig.savefig("./figures/" + self.config['model']['evaluate_res']+ "_"+self.config['model']['load_fine_tune_model'] + "_" + self.model_config + "_confusion_matrix_threshold_"+str(value)+"_"+time_now + ".png", dpi=fig.dpi)
            plt.close('all')