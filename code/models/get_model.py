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
                if args.old and len(args.from_checkpoint) > 4:
                    model = CG_model.from_pretrained(args)
                else:
                    model = CG_model(args)
            elif args.model == "QA":
                if args.old and len(args.from_checkpoint) > 4:
                    model = QA_model.from_pretrained(args)
                else:
                    model = QA_model(args)
            elif args.model == "QA+":
                if args.old and len(args.from_checkpoint) > 4:
                    model = QA_fast_model.from_pretrained(args)
                else:
                    model = QA_fast_model(args)
            else:
                raise ValueError(f"Model {args.model} not recognized")
            break
        except Exception as e:
            print(e)
    
    if not args.old and len(args.from_checkpoint) > 4:
        model.load_state_dict(torch.load(args.from_checkpoint, map_location=args.device))

    return model
