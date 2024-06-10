# includes: tokenizer, (m)T5, lama, but do lama later

from transformers import MT5ForConditionalGeneration, MT5ForQuestionAnswering

def get_model(args):
    # Load the model either from scratch or from a checkpoint

    if args.use_QA_model:
        model = MT5ForQuestionAnswering.from_pretrained(args.backbone)
    else:
        model = MT5ForConditionalGeneration.from_pretrained(args.backbone)
    return model
