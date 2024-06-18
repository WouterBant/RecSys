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

```bash
usage: train.py [-h] [--backbone BACKBONE] [--tokenizer TOKENIZER]
                [--from_checkpoint FROM_CHECKPOINT] [--lr LR] [--labda LABDA]
                [--datafraction DATAFRACTION] [--T T] [--n_epochs N_EPOCHS]
                [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
                [--current_step CURRENT_STEP] [--warmup_steps WARMUP_STEPS]
                [--old] [--debug] [--evaltrain] [--use_wandb]
                [--dataset DATASET] [--model {QA,QA+,CG}]
                [--prompt {titles,subtitles,QA+,diversity,pubtime}]

options:
  -h, --help            show this help message and exit
  --backbone BACKBONE   backbone model
  --tokenizer TOKENIZER
                        tokenizer model
  --from_checkpoint FROM_CHECKPOINT
                        load model from checkpoint
  --lr LR               learning rate
  --labda LABDA         lambda for pairwise ranking loss
  --datafraction DATAFRACTION
                        fraction of data to use
  --T T                 number of previous clicked articles to include in the
                        prompt
  --n_epochs N_EPOCHS   number of epochs
  --batch_size BATCH_SIZE
                        batch size
  --num_workers NUM_WORKERS
                        number of workers
  --current_step CURRENT_STEP
                        starting step for cosine learning rate
  --warmup_steps WARMUP_STEPS
                        number of warmup steps
  --old                 old way of loading from pretrained model
  --debug               debug mode
  --evaltrain           for evaluating on training set
  --use_wandb           Use Weights and Biases for logging
  --dataset DATASET     dataset to train on
  --model {QA,QA+,CG}   model to train
  --prompt {titles,subtitles,QA+,diversity,pubtime}
```


For overfitting on a small dataset, you can use:

```bash
python train.py --debug --T 4 --lr 0.001 --batch_size 16 --labda 0.0 --n_epochs 10000 --dataset demo --datafraction 0.001 --n_epochs 10000 --warmup_steps 500 --model [QA/QA+/CG] --prompt [titles/subtitles/QA+/diversity/pubtime]
```

For evaluating a pretrained model:
```bash
python evaluate.py
```

For creating a submission file with a pretrained model:
```bash
python create_submission_file.py
```

## Notebooks

In the [notebooks](notebooks) directory we provide notebooks for data preparation and debugging.

## Acknowledgement 
The approach largely follows the paper [PBNR: Prompt-based News Recommender System](https://arxiv.org/abs/2304.07862). However, we chose to write the code from scratch as we found this easier as opposed to getting the provided code to work. Thus our implementation differs from this paper but the idea is the same.


## Start of predictions file where 0's start (delete later)
0 [127,156,17,241
