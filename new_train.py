import json
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset

# === Ayarlar === 
basePath="C:\\PythonProject\\Github Project\\NLP_Project"
modelPath="C:\\PythonProject\\Github Project\\NLP_Project\\model"
intentPath="C:\\PythonProject\\Github Project\\NLP_Project\\intents.json"

# === Veriyi Yükle ===
with open(intentPath, "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

# === Label Encoding ===
le = LabelEncoder()
labels_enc = le.fit_transform(labels)

# === Tokenizer ve Dataset ===
tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
encodings = tokenizer(sentences, truncation=True, padding=True)

dataset = Dataset.from_dict({
    "input_ids": encodings["input_ids"],
    "attention_mask": encodings["attention_mask"],
    "labels": labels_enc
})

# === Eğitim / Doğrulama Bölme ===
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# === Model Oluştur ===
model = BertForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-turkish-cased",
    num_labels=len(le.classes_)
)

# === Eğitim Ayarları ===
training_args = TrainingArguments(
    output_dir=modelPath,
    eval_strategy="epoch",
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_total_limit=1,
    logging_dir=f"{basePath}\\logs",
    logging_steps=10
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# === Eğitimi Başlat ===
trainer.train()

# === Test Verisinde Değerlendir ===
predictions = trainer.predict(eval_dataset)
preds = torch.argmax(torch.tensor(predictions.predictions), dim=1)

# === Doğruluk Hesapla ===
acc = accuracy_score(eval_dataset["labels"], preds)
print(f"\n Accuracy (Doğruluk): {acc:.2%}")

# === Confusion Matrix Çiz ===
cm = confusion_matrix(eval_dataset["labels"], preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# === Model ve Etiketleri Kaydet ===
os.makedirs(modelPath, exist_ok=True)
model.save_pretrained(modelPath, safe_serialization=False)
tokenizer.save_pretrained(modelPath)
with open(f"{modelPath}/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
