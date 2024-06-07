# # Use a pipeline as a high-level helper
# from transformers import pipeline

# # pipe = pipeline("question-answering", model="google/mt5-small")

# # out = pipe(question="kommer jag att gilla den här artikeln: 'Fotbollsrecap sverige'. svara med 'ja' eller 'nej'", 
# #            context="Jag har tidigare läst och gillat artiklar om sport")
# # print(out)

# pipe = pipeline("text2text-generation", model="google/mt5-small")
# out = pipe("kommer jag att gilla den här artikeln: 'Fotbollsrecap sverige'. svara med 'ja' eller 'nej'")
# print(out)

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
# model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

# model.train()

# # article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
# article = "dansk javlar perkele."

# # summary = "Weiter Verhandlung in Syrien."

# inputs = tokenizer(article, text_target=article, return_tensors="pt")

# print(inputs)

# outputs = model.base_model(**inputs)
# # print(outputs)

# loss = outputs.loss
# print(loss)

# loss.backward()

# model.eval()

# input_ids = tokenizer(

#     "dansk javlar blabalballa perkele", return_tensors="pt"

# ).input_ids  # Batch size 1

# outputs = model.generate(input_ids)

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

print(tokenizer("yes"))
print(tokenizer("no"))

# Sample input text
input_text = "yes yes yes yes yes yes yes"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt")
print(inputs)

# Prepare decoder input ids (usually the start token)
decoder_input_ids = tokenizer("<pad>", return_tensors="pt").input_ids

# Forward pass with decoder input ids
outputs = model.base_model(input_ids=inputs["input_ids"], decoder_input_ids=decoder_input_ids)

# Extract the logits (if needed)
logits = outputs.logits
print(logits.shape)

print('logit yes', outputs.logits[0,0,36339])
print('logit no',  outputs.logits[0,0,375])
print('logit yes', outputs.logits[0,1,36339])
print('logit no',  outputs.logits[0,1,375])

print(tokenizer.decode(outputs.logits[0,1].argmax()), outputs.logits[0,1].max())

from torch.nn import CrossEntropyLoss

ce = CrossEntropyLoss()

# nll_loss = ce()


import torch

def compute_rank_loss(logits_pos, logits_neg):
    r_pos = torch.sigmoid(logits_pos)
    r_neg = torch.sigmoid(logits_neg)
    diff = torch.sigmoid(r_pos - r_neg)
    return torch.log(1e-8 + torch.exp(diff))

lamb=0.5

# loss = lamb * nll_loss + (1-lamb) * compute_rank_loss(l)
