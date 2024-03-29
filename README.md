# LLM_fine_tuning_direct
In this project, I fine-tuned Flan-t5 models (large, XL) by implementing two algorithms, supervised fine-tuning (SFT) and [proximal policy optimization](https://arxiv.org/pdf/1707.06347.pdf) (PPO), a reinforcement learning from human feedback (RLHF) algorithm using [trl](https://huggingface.co/docs/trl/index) library on a risk characteristic QA task. 

### Reinforcement Learning from Human Feedback

<img src="rlhf.png" alt="RLHF" width="400"/>

[More](https://huggingface.co/blog/rlhf) to read

#### install env
```
conda create -n LLM_FT_direct_env python=3.10.4
```
#### activate env
```
conda activate LLM_FT_direct_env 
```
#### clone and install
```[requirements.txt](requirements.txt)
git clone https://github.com/xliu-522/LLM_fine_tuning_direct.git
cd LLM_fine_tuning_direct
pip install slam-llms
pip install -r requirements.txt
