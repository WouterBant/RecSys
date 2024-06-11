# includes: tokenizer, (m)T5, lama, but do lama later

from transformers import MT5ForConditionalGeneration, MT5ForQuestionAnswering
import torch


def get_model(args):
    # Load the model either from scratch or from a checkpoint
    # Sometimes cannot connect to Huggingface, so we try multiple times
    for _ in range(10):
        try:
            if args.use_QA_model:
                model = MT5ForQuestionAnswering.from_pretrained(args.backbone)
            else:
                model = MT5ForConditionalGeneration.from_pretrained(args.backbone)
            break
        except:
            pass
    
    if len(args.from_checkpoint) > 4:
        model.load_state_dict(torch.load(args.from_checkpoint))

    return model
