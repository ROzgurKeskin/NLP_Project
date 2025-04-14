import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset 

modelPath="C:\\PythonProject\\Github Project\\UskudarProject\\model"
intentPath="C:\\PythonProject\\Github Project\\UskudarProject\\intents.json"

# Veriyi yükle
with open(intentPath, "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

# Etiket kodlama
le = LabelEncoder()
labels_enc = le.fit_transform(labels)

# Tokenizer ve Dataset
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

encodings = tokenizer(sentences, truncation=True, padding=True)
dataset = Dataset.from_dict({
    "input_ids": encodings["input_ids"],
    "attention_mask": encodings["attention_mask"],
    "labels": labels_enc
})

# Eğitim/Doğrulama bölme
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Model
model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased",num_labels=len(le.classes_))

# Eğitim ayarları
training_args = TrainingArguments(
    output_dir=modelPath,
    eval_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_total_limit=1,
    logging_dir="C:\\PythonProject\\Github Project\\UskudarProject\\logs",
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

# Model ve etiketleri kaydet
model.save_pretrained(modelPath)
tokenizer.save_pretrained(modelPath,safe_serialization=False)

with open(f"{modelPath}\\label_encoder.pkl", "wb") as f:
    import pickle
    pickle.dump(le, f)