# Ranking Danish news articles with multilingual T5

## About

Inspired by [PBNR: Prompt-based News Recommender System](https://arxiv.org/abs/2304.07862) we cast news recommendation as a text-to-text generation problem. We use click logs from a Danish news site provided in the [RecSys Challenge 2024](https://www.recsyschallenge.com/2024/).

## Getting started

### Clone code to your device

```bash
git clone https://github.com/WouterBant/RecSys.git
```

```bash
cd RecSys
```

### Environment

For managing dependencies we use Conda, see https://conda.io/projects/conda/en/latest/user-guide/install/index.html for installation instructions. When installed take the following steps:

```bash
conda env create -f environment.yml
```

```bash
conda activate RecSys
```

### Running experiments  TODO update the instructies later 

> Note that the relevant data will be downloaded automatically without asking for permission when running the following commands. Also, our code uses GPUs when available by default but also works on CPUs.

First change directory to the folder containing all the code:

```bash
cd code
```

We provide many options for running the code:

```
usage: train.py [-h] [--backbone BACKBONE] [--tokenizer TOKENIZER]
                [--from_checkpoint FROM_CHECKPOINT] [--lr LR] [--labda LABDA]
                [--datafraction DATAFRACTION] [--T T] [--n_epochs N_EPOCHS]
                [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
                [--current_step CURRENT_STEP] [--warmup_steps WARMUP_STEPS]
                [--old] [--debug] [--evaltrain] [--use_wandb]
                [--dataset DATASET] [--model {QA,QA+,CG,CGc}]
                [--prompt {titles,subtitles,QA+,diversity,pubtime}]

options:
  -h, --help                            show this help message and exit
  --backbone BACKBONE                   backbone model
  --tokenizer TOKENIZER                 tokenizer model
  --from_checkpoint FROM_CHECKPOINT     load model from checkpoint
  --lr LR                               learning rate
  --labda LABDA                         lambda for pairwise ranking loss
  --datafraction DATAFRACTION           fraction of data to use
  --T T                                 number of previous clicked articles to include in the prompt
  --n_epochs N_EPOCHS                   number of epochs
  --batch_size BATCH_SIZE               batch size
  --num_workers NUM_WORKERS             number of workers
  --current_step CURRENT_STEP           starting step for cosine learning rate
  --warmup_steps WARMUP_STEPS           number of warmup steps
  --old                                 old way of loading from pretrained model
  --debug                               debug mode
  --evaltrain                           for evaluating on training set
  --use_wandb                           Use Weights and Biases for logging
  --dataset DATASET                     dataset to train on
  --model {QA,QA+,CG, CGc}              model to train
  --prompt {titles,subtitles,QA+,diversity,pubtime}
```

For overfitting on a small dataset, you can use:

```bash
python train.py --debug --T 4 --lr 0.001 --batch_size 16 --labda 0.0 --n_epochs 10000 --dataset demo --datafraction 0.001 --n_epochs 10000 --warmup_steps 500 --model [QA/QA+/CG/CGc] --prompt [titles/subtitles/QA+/diversity/pubtime]
```

We ran many experiments, see [main.sh](main.sh) for all commands we used. Uncomment the ones you want to reproduce. Subsequently run:

```bash
./main.sh
```

Note that we have Weights and Biases integration which can be used with the ```--use_wandb```  flag.

## Checkpoint

We provide the checkpoint from the model that we used to obtain the results we submitted: https://drive.google.com/file/d/1PpdwiHut4mmPB363HiNJVmOEcZsbTB5l/view?usp=sharing 


After placing this checkpoint in the checkpoints folder, you can evaluate this model with:

TODO: make this simpler
```bash
python evaluate.py --from_checkpoint ../../../checkpoint/model.pth
```

And for creating a submission file with this model:
```bash
python create_submission_file.py
```