#!/bin/bash

cd code

# We recommend not to run this script as is as it will take a long time to train all the models.
# Instead, you can run the following commands one by one in the terminal.
# Also, as mentioned in the paper, we don't train longer than 8 hours and earlier when no further improvement is observed.

labda_values=(0.0 0.4)
model_values=(QA CG)
prompt_values=(titles subtitles diversity pubtime)

for labda in "${labda_values[@]}"; do
    for model in "${model_values[@]}"; do
        for prompt in "${prompt_values[@]}"; do
            python train.py --batch_size 32 --labda $labda --dataset large --model $model --prompt $prompt
        done
    done
done

python train.py --batch_size 16 --labda 0.0 --dataset large --model QA+ --prompt QA+
python train.py --batch_size 16 --labda 0.4 --dataset large --model QA+ --prompt QA+