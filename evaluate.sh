#!/bin/bash

log_error() {
    echo "$(date) - Error in script at line $1: $2" >> error_log.txt
}

trap 'log_error $LINENO "$BASH_COMMAND"' ERR


# ----------------------------------------------------------------------------------
# It's important to note that:
# - evaluation takes around 10 minutes on a L4 GPU (availabe on lightning.ai)
# - the baselines don't require any checkpoints other than the automatically donwloaded mT5 checkppoint
# - other runs require checkpoints that can be downloaded the line before the python command
# - the checkpoints of the CG model are 1.2GB and for the other models this is 657MB
# - batch size 1 is required for evaluation
# ----------------------------------------------------------------------------------


cd code

### Baselines (don't require further checkpoints):
# python evaluate.py --model CG --dataset demo --evaltrain --prompt titles
# python evaluate.py --model CG --dataset demo --prompt titles

# python evaluate.py --model CG --dataset demo --evaltrain --prompt subtitles
# python evaluate.py --model CG --dataset demo --prompt subtitles

# python evaluate.py --model CG --dataset demo --evaltrain --prompt diversity
# python evaluate.py --model CG --dataset demo --prompt diversity

# python evaluate.py --model CG --dataset demo --evaltrain --prompt pubtime
# python evaluate.py --model CG --dataset demo --prompt pubtime


### Evaluating the trained models
# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.0_model_CGc_prompt_titles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt titles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_CGc_prompt_titles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt titles --evaltrain ../checkpoints/model_lr_0.0001_lab_0.0_model_CGc_prompt_titles.pth

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.4_model_CGc_prompt_titles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt titles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CGc_prompt_titles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt titles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CGc_prompt_titles.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.0_model_CGc_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt subtitles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_CGc_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt subtitles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_CGc_prompt_subtitles.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.4_model_CGc_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt subtitles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CGc_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt subtitles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CGc_prompt_subtitles.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.0_model_CGc_prompt_diversity.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt diversity --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_CGc_prompt_diversity.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt diversity --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_CGc_prompt_diversity.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.4_model_CGc_prompt_diversity.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt diversity --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CGc_prompt_diversity.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt diversity --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CGc_prompt_diversity.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.0_model_CGc_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt pubtime --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_CGc_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt pubtime --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_CGc_prompt_pubtime.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.4_model_CGc_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt pubtime --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CGc_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --dataset demo --model CGc --prompt pubtime --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CGc_prompt_pubtime.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.0_model_CG_prompt_titles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt titles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_CG_prompt_titles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt titles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_CG_prompt_titles.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.4_model_CG_prompt_titles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt titles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CG_prompt_titles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt titles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CG_prompt_titles.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.0_model_CG_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt subtitles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_CG_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt subtitles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_CG_prompt_subtitles.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.4_model_CG_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt subtitles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CG_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt subtitles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CG_prompt_subtitles.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.0_model_CG_prompt_diversity.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt diversity --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_CG_prompt_diversity.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt diversity --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_CG_prompt_diversity.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.4_model_CG_prompt_diversity.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt diversity --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CG_prompt_diversity.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt diversity --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CG_prompt_diversity.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.0_model_CG_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt pubtime --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_CG_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt pubtime --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_CG_prompt_pubtime.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.4_model_CG_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt pubtime --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CG_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --dataset demo --model CG --prompt pubtime --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_CG_prompt_pubtime.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.0_model_QA_prompt_titles.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt titles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_QA_prompt_titles.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt titles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_QA_prompt_titles.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.4_model_QA_prompt_titles.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt titles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_QA_prompt_titles.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt titles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_QA_prompt_titles.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.0_model_QA_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt subtitles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_QA_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt subtitles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_QA_prompt_subtitles.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.4_model_QA_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt subtitles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_QA_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt subtitles --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_QA_prompt_subtitles.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.0_model_QA_prompt_diversity.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt diversity --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_QA_prompt_diversity.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt diversity --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_QA_prompt_diversity.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.4_model_QA_prompt_diversity.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt diversity --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_QA_prompt_diversity.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt diversity --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_QA_prompt_diversity.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.0_model_QA_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt pubtime --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_QA_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt pubtime --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.0_model_QA_prompt_pubtime.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_0.0001_lab_0.4_model_QA_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt pubtime --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_QA_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA --prompt pubtime --from_checkpoint ../checkpoints/model_lr_0.0001_lab_0.4_model_QA_prompt_pubtime.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_5e-05_lab_0.0_model_QA+_prompt_QA+.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA+ --prompt QA+ --from_checkpoint ../checkpoints/model_lr_5e-05_lab_0.0_model_QA+_prompt_QA+.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA+ --prompt QA+ --from_checkpoint ../checkpoints/model_lr_5e-05_lab_0.0_model_QA+_prompt_QA+.pth --evaltrain

# wget --header="Referer: https://huggingface.co/" -P ../checkpoints https://huggingface.co/Wouter01/mT5Ranking/resolve/main/model_lr_5e-05_lab_0.4_model_QA+_prompt_QA+.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA+ --prompt QA+ --from_checkpoint ../checkpoints/model_lr_5e-05_lab_0.4_model_QA+_prompt_QA+.pth
# python evaluate.py --batch_size 1 --dataset demo --model QA+ --prompt QA+ --from_checkpoint ../checkpoints/model_lr_5e-05_lab_0.4_model_QA+_prompt_QA+.pth --evaltrain