#!/bin/bash

python evaluate.py --model CG --dataset demo --evaltrain --prompt titles
python evaluate.py --model CG --dataset demo --prompt titles

python evaluate.py --model CG --dataset demo --evaltrain --prompt subtitles
python evaluate.py --model CG --dataset demo --prompt subtitles

python evaluate.py --model CG --dataset demo --evaltrain --prompt diversity
python evaluate.py --model CG --dataset demo --prompt diversity

python evaluate.py --model CG --dataset demo --evaltrain --prompt pubtime
python evaluate.py --model CG --dataset demo --prompt pubtime
