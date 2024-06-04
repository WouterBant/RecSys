## Clean code

- dont use inheritance when the base class is only used once
- write comments use whitelines
- do everything in the most simple way possible while minimizing the lines of code
- make sure everything also works when using multiple gpus
- use their code as starting point, but simplify and remove bugs

### order for implementation
1. dataloader
2. models
3. prompt_templates
3. train
4. metrics
5. evaluate
