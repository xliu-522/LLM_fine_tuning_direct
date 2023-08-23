import os
import pandas as pd

class Data(object):
    def __init__(self, config):
        self.file_name = config['data']['data_file_name']
        self.row_limit = config['data']['row_limit']
        self.base_model_name = config['model']['base_model_name']
        self.base_model_review = config['model']['base_model_review']
        self.config = config

    def read(self):
        my_data = pd.read_csv('data/'+self.file_name)
        my_data['labels'] = my_data['labels'].astype('int')
        my_data = my_data.head(self.row_limit)
        self.my_data = my_data[["Account_Name", "Question", self.base_model_review, "labels", "q_positive", "q_negative", "impossible to draw a conclusion"]]
        #self.my_data.rename(columns={self.base_model_name: "model_name"}, errors="raise", inplace=True)

    def prompt_eng(self):
        split_name = self.base_model_name.split('_', 1)
        language_model_name = split_name[0]
        if language_model_name in ["lg", "large"]:
            self.language_model_name = "flan-t5-large"
        elif language_model_name in ["xl"]:
            self.language_model_name = "flan-t5-xl"
        elif language_model_name in ["xxl"]:
            self.language_model_name = "flan-t5-xxl"
        else:
            print("Warning: language_model_name part of base_model_name is not known: ", self.base_model_name)

        prompt_name = split_name[1]

        if "name" in prompt_name:
            self.tag_translation = self.config['model']['tag_translation_no_nli']
            #self.my_data["prompt"] = "Context: This is a customer review for " + self.my_data["Account_Name"] + ". " + self.my_data["Review"] + "." + self.my_data["Question"] + "\n\nOptions:\n- Yes\n- No\n- Undecided\n\n"
        elif "noname" in prompt_name:
            self.tag_translation = self.config['model']['tag_translation_no_nli']
            #self.my_data["prompt"] = "Context: This is a customer review - "+self.my_data["Review"]+". "+self.my_data["Question"]+"\n\nOptions:\n- Yes\n- No\n- Undecided\n\n"
        elif "nli" in prompt_name:
            self.tag_translation = self.config['model']['tag_translation_nli']
            #self.my_data["prompt"] = self.my_data["Review"]+".\nBased on the paragraph above, can we conclude that\n\na)"+self.my_data["q_positive"]+"\nb)"+self.my_data["q_negative"]+"\nc)"+self.my_data["impossible to draw a conclusion"]+"\n\n"
        else:
            print("Warning: prompt_name part of base_model_name is not known: ", self.base_model_name)
        prompt_list = []
        for ii, row in self.my_data.iterrows():
            google_name = row["Account_Name"]
            review = row[self.base_model_review]
            question = row["Question"]
            q_positive = row["q_positive"]
            q_negative = row["q_negative"]
            if "name" in prompt_name:
                prompt_list.append(f"""Context: This is a customer review for {google_name}. "{review}". {question}\n\nOptions:\n- Yes\n- No\n- Undecided\n\n """)
            elif "noname" in prompt_name:
                prompt_list.append(f"""Context: This is a customer review - "{review}". {question}\n\nOptions:\n- Yes\n- No\n- Undecided\n\n """)
            elif "nli" in prompt_name:
                prompt_list.append(f""""{review}".\nBased on the paragraph above, can we conclude that\n\na){q_positive}\nb){q_negative}\nc)impossible to draw a conclusion\n\n """)
        self.my_data["prompt"] = prompt_list