## PGNR
Source code for the paper 'The Prompt-based Generative News Recommender System' 


## TODO
modellen vanuit checkpoint laden overal waar we nu een model laden als optie

## List of special tokens

- -100: padding token
- 150: ?
- 4273: ?

## Acknowledgement 
The overall structure of the code is derived from the open-source P5 project (https://github.com/jeykigung/P5). We appreciate their significant contribution to the research community.


0 [127,156,17,241


#quick run for overfit:
python train.py --use_wandb --debug --T 4 --lr 0.001 --batch_size 16 --labda 0.0 --dataset demo --datafraction 0.001 --n_epochs 10000 --use_QA_model


torch.tensor([[-0.9477, -3.7398, -1.0754, -0.7692, -4.1885],        [-1.1750, -0.6564, -1.8939, -1.5410, -1.9851],        [-2.4754, -2.3068, -1.5064, -1.7220, -0.7567],        [-0.8613, -0.7153, -2.3804,  0.0215, -9.8556],        [-1.0081, -0.9542, -1.9504, -1.5692, -1.8795],[-2.6071, -3.2940, -2.0377, -2.3994, -2.0357],[-1.1059, -1.5805, -1.7454, -1.8936, -1.5744],[-2.1765, -2.8839, -1.4926, -4.0404, -3.1123]])