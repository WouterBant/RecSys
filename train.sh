#!/bin/bash

log_error() {
    echo "$(date) - Error in script at line $1: $2" >> error_log.txt
}

trap 'log_error $LINENO "$BASH_COMMAND"' ERR


# ----------------------------------------------------------------------------------
# It's important to note that:
# - we trained all models for approximately 8 hours on a single A100 GPU with 40GB RAM
# - checkpoints for each of the models below are already available at  TODO insert url
# - uncommenting all of the below will results in a run time of at least a month on a single A100 GPU (on the large dataset)
# - if you decide to increase/decrease the batch size we recommend changing the learning rate by the same factor
# ----------------------------------------------------------------------------------


cd code

# python train.py --batch_size 32 --labda 0.0 --dataset large --model CGc --prompt titles
# python train.py --batch_size 32 --labda 0.4 --dataset large --model CGc --prompt titles

# python train.py --batch_size 32 --labda 0.0 --dataset large --model CGc --prompt subtitles
# python train.py --batch_size 32 --labda 0.4 --dataset large --model CGc --prompt subtitles

# python train.py --batch_size 32 --labda 0.0 --dataset large --model CGc --prompt diversity
# python train.py --batch_size 32 --labda 0.4 --dataset large --model CGc --prompt diversity

# python train.py --batch_size 32 --labda 0.0 --dataset large --model CGc --prompt pubtime
# python train.py --batch_size 32 --labda 0.4 --dataset large --model CGc --prompt pubtime

# python train.py --batch_size 32 --labda 0.0 --dataset large --model CG --prompt titles
# python train.py --batch_size 32 --labda 0.4 --dataset large --model CG --prompt titles

# python train.py --batch_size 32 --labda 0.0 --dataset large --model CG --prompt subtitles
# python train.py --batch_size 32 --labda 0.4 --dataset large --model CG --prompt subtitles

# python train.py --batch_size 32 --labda 0.0 --dataset large --model CG --prompt diversity
# python train.py --batch_size 32 --labda 0.4 --dataset large --model CG --prompt diversity

# python train.py --batch_size 32 --labda 0.0 --dataset large --model CG --prompt pubtime
# python train.py --batch_size 32 --labda 0.4 --dataset large --model CG --prompt pubtime

# python train.py --batch_size 16 --labda 0.0 --dataset large --model QA --prompt titles
# python train.py --batch_size 16 --labda 0.4 --dataset large --model QA --prompt titles

# python train.py --batch_size 16 --labda 0.0 --dataset large --model QA --prompt subtitles
# python train.py --batch_size 16 --labda 0.4 --dataset large --model QA --prompt subtitles

# python train.py --batch_size 16 --labda 0.0 --dataset large --model QA --prompt diversity
# python train.py --batch_size 16 --labda 0.4 --dataset large --model QA --prompt diversity

# python train.py --batch_size 16 --labda 0.0 --dataset large --model QA --prompt pubtime
# python train.py --batch_size 16 --labda 0.4 --dataset large --model QA --prompt pubtime