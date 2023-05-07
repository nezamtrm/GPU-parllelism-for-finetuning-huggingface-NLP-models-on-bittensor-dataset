# GPU parallelism for finetuning three different huggingface NLP models on Bittensor dataset

In this repo, the Bittensor CLM finetuning script is converted to support multi gpus. The bittensor version of the script has been adapted from Hugging Face's transformers/language-modeling code and can be found here: https://github.com/opentensor/clm_model_tuning

This script works with 2/4/6/8 GPUs in parallel to be able to train larger models or train them faster. It is written to be trained on the bittensor dataset and work for the huggingface models as gpt-neo-2.7B, gpt-j-6B and gpt-neeo-1.3B

## Dataset

The information about the dataset is found https://docs.bittensor.com/nested/TheDataset.html

## How to Run

Run git clone https://github.com/opentensor/clm_model_tuning.git
Replace the content of finetune_using_clm2.py with finetune_using_clm2.py in the reference repo
Run each notebook to finetune the gpt-neo-2.7B, gpt-j-6B and gpt-neeo-1.3B models on bittensor dataset.
