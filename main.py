import argparse
import json
from src.data import Data
from src.ppo_fine_tune import PPO_Fine_Tune
from src.sft_fine_tune import SFT_Fine_Tune
from src.evaluate import Evaluate
from unique_names_generator import get_random_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs='?', type=str, default="config/config.json", required=False)
    args = parser.parse_args()
    try:
        with open(args.config, ) as config:
            config = json.load(config)
            config['model']["random_name"] = get_random_name().replace(" ", "_")
    except:
        print("Error in config")

    print("**** Reading data ****")
    data_obj = Data(config)
    data_obj.read()

    print("**** prompt engineering ****")
    data_obj.prompt_eng()

    if config['model']['eval_FineTune'] == "FineTune":
        if config['model']['fine_tuning_algo'] == 'PPO':
            print("**** PPO Fine tuning started ****")
            FineTune_obj = PPO_Fine_Tune(config=config, data=data_obj.my_data, language_model_name=data_obj.language_model_name, tag_translation = data_obj.tag_translation)
            FineTune_obj.train_it()
        elif config['model']['fine_tuning_algo'] == 'SFT':
            print("**** SFT Fine tuning started ****")
            FineTune_obj = SFT_Fine_Tune(config=config, data=data_obj.my_data, language_model_name=data_obj.language_model_name, tag_translation = data_obj.tag_translation)
            FineTune_obj.train_it()
    else:
        print("**** Evaluation started ****")

        Evaluate_obj = Evaluate(config=config, data=data_obj.my_data, language_model_name=data_obj.language_model_name, tag_translation = data_obj.tag_translation)
        if config['model']['evaluate_res'] == 'new':
            Evaluate_obj.infer()

        Evaluate_obj.metrics()

    print("Done!")

if __name__ == "__main__":
    main()