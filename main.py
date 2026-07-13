import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "model/ruroberta-sentiment-5class"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

text = "Зарплату задерживают, начальник хамит, работать невозможно."

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

id2label = model.config.id2label

for i, p in enumerate(probs):
    print(id2label[i], float(p))

print("decision:", id2label[int(probs.argmax())])
