

export CUDA_VISIBLE_DEVICES=0
name=MIND-eval

output=snap/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
        --master_port 1455\
        ./src/evaluate.py \
        --distributed --multiGPU \
        --seed 3407 \
        --load  # load checkpoint here \
        --test MIND \
        --val_batch_size 2000\
        --backbone 't5-small' \
        --output $output ${@:2} \
        --max_text_length 512\
        --gen_max_length 64 \
        --history_length 5 \
        --pair_weight 0.4\
        --whole_word_embed > $name.log
