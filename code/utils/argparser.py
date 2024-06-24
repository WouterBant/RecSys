import argparse


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='google/mt5-small', help='backbone model')
    parser.add_argument('--tokenizer', type=str, default='google/mt5-small', help='tokenizer model')
    parser.add_argument('--from_checkpoint', type=str, default='', help='load model from checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--labda', type=float, default=0.4, help='lambda for pairwise ranking loss')
    parser.add_argument('--datafraction', type=float, default=1.0, help='fraction of data to use')
    parser.add_argument('--T', type=int, default=4, help='number of previous clicked articles to include in the prompt')
    parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--current_step', type=int, default=0, help='starting step for cosine learning rate')
    parser.add_argument('--warmup_steps', type=int, default=15000, help='number of warmup steps')
    parser.add_argument('--old', action='store_true', help='old way of loading from pretrained model')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--evaltrain', action='store_true', help='for evaluating on training set')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    parser.add_argument('--dataset', type=str, default='demo', help='dataset to train on')
    parser.add_argument('--model', type=str, choices=["QA", "QA+", "CG", "CGc"], default='CG', help='model to train')
    parser.add_argument('--prompt', type=str, choices=["titles", "subtitles", "QA+", "diversity", "pubtime"], default='titles', help='prompt to use')
    args = parser.parse_args()
    print(args)
    return args