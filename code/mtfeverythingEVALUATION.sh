#!/bin/bash

log_error() {
    echo "$(date) - Error in script at line $1: $2" >> error_log.txt
}

trap 'log_error $LINENO "$BASH_COMMAND"' ERR

# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CGc --prompt titles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CGc_prompt_titles.pth
# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CGc --prompt titles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CGc_prompt_titles.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CGc --prompt titles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CGc_prompt_titles.pth
# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CGc --prompt titles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CGc_prompt_titles.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CGc --prompt subtitles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CGc_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CGc --prompt subtitles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CGc_prompt_subtitles.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CGc --prompt subtitles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CGc_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CGc --prompt subtitles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CGc_prompt_subtitles.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CGc --prompt diversity --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CGc_prompt_diversity.pth
# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CGc --prompt diversity --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CGc_prompt_diversity.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CGc --prompt diversity --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CGc_prompt_diversity.pth
# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CGc --prompt diversity --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CGc_prompt_diversity.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CGc --prompt pubtime --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CGc_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CGc --prompt pubtime --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CGc_prompt_pubtime.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CGc --prompt pubtime --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CGc_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CGc --prompt pubtime --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CGc_prompt_pubtime.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CG --prompt titles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CG_prompt_titles.pth
# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CG --prompt titles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CG_prompt_titles.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CG --prompt titles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CG_prompt_titles.pth
# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CG --prompt titles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CG_prompt_titles.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CG --prompt subtitles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CG_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CG --prompt subtitles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CG_prompt_subtitles.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CG --prompt subtitles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CG_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CG --prompt subtitles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CG_prompt_subtitles.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CG --prompt diversity --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CG_prompt_diversity.pth
# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CG --prompt diversity --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CG_prompt_diversity.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CG --prompt diversity --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CG_prompt_diversity.pth
# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CG --prompt diversity --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CG_prompt_diversity.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CG --prompt pubtime --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CG_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model CG --prompt pubtime --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_CG_prompt_pubtime.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CG --prompt pubtime --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CG_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model CG --prompt pubtime --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_CG_prompt_pubtime.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model QA --prompt titles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_QA_prompt_titles.pth
# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model QA --prompt titles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_QA_prompt_titles.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model QA --prompt titles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_QA_prompt_titles.pth
# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model QA --prompt titles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_QA_prompt_titles.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model QA --prompt subtitles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_QA_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model QA --prompt subtitles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_QA_prompt_subtitles.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model QA --prompt subtitles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_QA_prompt_subtitles.pth
# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model QA --prompt subtitles --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_QA_prompt_subtitles.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model QA --prompt diversity --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_QA_prompt_diversity.pth
# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model QA --prompt diversity --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_QA_prompt_diversity.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model QA --prompt diversity --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_QA_prompt_diversity.pth
# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model QA --prompt diversity --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_QA_prompt_diversity.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model QA --prompt pubtime --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_QA_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --labda 0.0 --dataset demo --model QA --prompt pubtime --from_checkpoint ../../../new/model_lr_0.0001_lab_0.0_model_QA_prompt_pubtime.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model QA --prompt pubtime --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_QA_prompt_pubtime.pth
# python evaluate.py --batch_size 1 --labda 0.4 --dataset demo --model QA --prompt pubtime --from_checkpoint ../../../new/model_lr_0.0001_lab_0.4_model_QA_prompt_pubtime.pth --evaltrain

python evaluate.py --batch_size 1 --labda 5e-5 --dataset demo --model QA+ --prompt QA+ --from_checkpoint ../../../new/model_lr_5e-05_lab_0.0_model_QA+_prompt_QA+.pth
python evaluate.py --batch_size 1 --labda 5e-5 --dataset demo --model QA+ --prompt QA+ --from_checkpoint ../../../new/model_lr_5e-05_lab_0.0_model_QA+_prompt_QA+.pth --evaltrain

# python evaluate.py --batch_size 1 --labda 5e-5 --dataset demo --model QA+ --prompt QA+ --from_checkpoint ../../../new/model_lr_5e-05_lab_0.4_model_QA+_prompt_QA+.pth
# python evaluate.py --batch_size 1 --labda 5e-5 --dataset demo --model QA+ --prompt QA+ --from_checkpoint ../../../new/model_lr_5e-05_lab_0.4_model_QA+_prompt_QA+.pth --evaltrain