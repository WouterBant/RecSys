import torch
from .cg_model import CG_model
from .qa_model import QA_model
from .qa_fast_model import QA_fast_model


def get_model(args):
    # Load the model either from scratch or from a checkpoint
    # Sometimes cannot connect to Huggingface, so we try multiple times
    for _ in range(10):
        try:
            if args.model == "CG":
                model = CG_model(args)
            elif args.model == "QA":
                model = QA_model(args)
            elif args.model == "QA+":
                model = QA_fast_model(args)
            else:
                raise ValueError(f"Model {args.model} not recognized")
        except:
            pass
    
    if len(args.from_checkpoint) > 4:
        model.load_state_dict(torch.load(args.from_checkpoint))

    return model
