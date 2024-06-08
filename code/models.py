# includes: tokenizer, (m)T5, lama, but do lama later

from transformers import MT5ForConditionalGeneration

def get_model(args):
    # Load the model either from scratch or from a checkpoint

    model = MT5ForConditionalGeneration.from_pretrained(args.backbone)
    return model
