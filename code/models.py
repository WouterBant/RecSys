# includes: tokenizer, (m)T5, lama, but do lama later

from transformers import AutoModelForSeq2SeqLM

def get_model(args):
    # Load the model either from scratch or from a checkpoint

    model = AutoModelForSeq2SeqLM.from_pretrained(args.backbone)
    return model
