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

For overfitting on a small dataset:

```bash
python train.py --debug --T 4 --lr 0.001 --batch_size 16 --labda 0.0 --n_epochs 10000 --dataset demo --datafraction 0.001 --n_epochs 10000
```

```bash
python train.py
```

```bash
python evaluate.py
```

## Notebooks

In the [notebooks](notebooks) directory we provide notebooks for data preparation and debugging.

## Acknowledgement 
The approach largely follows the paper [PBNR: Prompt-based News Recommender System](https://arxiv.org/abs/2304.07862). However, we chose to write the code from scratch as we found this easier as opposed to getting the provided code to work. Thus our implementation differs from this paper but the idea is the same.


## Start of predictions file where 0's start (delete later)
0 [127,156,17,241
