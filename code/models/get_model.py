import torch

from .cg_model import CG_model
from .cg_classifier import CG_classifier_model
from .qa_model import QA_model
from .qa_fast_model import QA_fast_model


model_mapping = {
    "CG": CG_model,
    "CGc": CG_classifier_model,
    "QA": QA_model,
    "QA+": QA_fast_model
}

def get_model(args):
    """
    Load the model either from scratch or from a checkpoint.
    Sometimes cannot connect to Huggingface, so we try multiple times.
    """

    for _ in range(10):
        try:
            model_class = model_mapping.get(args.model)

            if args.old and len(args.from_checkpoint) > 4:
                model = model_class.from_pretrained(args)
            else:
                model = model_class(args)
            break
        except Exception as e:
            print(e)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.old and len(args.from_checkpoint) > 4:
        print(model.load_state_dict(torch.load(args.from_checkpoint, map_location=device)))

    return model
