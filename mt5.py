# # Use a pipeline as a high-level helper
# from transformers import pipeline

# # pipe = pipeline("question-answering", model="google/mt5-small")

# # out = pipe(question="kommer jag att gilla den här artikeln: 'Fotbollsrecap sverige'. svara med 'ja' eller 'nej'", 
# #            context="Jag har tidigare läst och gillat artiklar om sport")
# # print(out)

# pipe = pipeline("text2text-generation", model="google/mt5-small")
# out = pipe("kommer jag att gilla den här artikeln: 'Fotbollsrecap sverige'. svara med 'ja' eller 'nej'")
# print(out)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

model.train()

# article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
article = "dansk javlar perkele."

# summary = "Weiter Verhandlung in Syrien."

inputs = tokenizer(article, text_target="dansk javlar", return_tensors="pt")

outputs = model(**inputs)
print(outputs)

loss = outputs.loss
print(loss)

loss.backward()

model.eval()

input_ids = tokenizer(

    "dansk javlar", return_tensors="pt"

).input_ids  # Batch size 1

outputs = model.generate(input_ids)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
