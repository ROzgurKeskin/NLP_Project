import json
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

from transformers import (
    BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments,
    EarlyStoppingCallback, get_scheduler
)
import torch
from torch.nn import CrossEntropyLoss
from datasets import Dataset

# === Ayarlar ===
basePath = "C:\\PythonProject\\Github Project\\NLP_Project"
modelPath = "C:\\PythonProject\\Github Project\\NLP_Project\\model"
modelFolder = "cross"
intentPath = "C:\\PythonProject\\Github Project\\NLP_Project\\intents.json"
resultSavingPath = "C:\\PythonProject\\Github Project\\NLP_Project\\Results\\images"
turkishModelPath = "dbmdz/bert-base-turkish-cased"
startTime = datetime.now()

# === Veriyi Yükle ===
with open(intentPath, "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])

# Sınıf Dağılımı
class_counts = Counter(labels)
print("Sınıf Dağılımı:", class_counts)

try:
    with open(f"{modelPath}/{modelFolder}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
except FileNotFoundError:
    le = LabelEncoder()

labels_enc = le.fit_transform(labels)

# Plot the class distribution
plt.bar(class_counts.keys(), class_counts.values())


# Tokenizer
tokenizer = BertTokenizer.from_pretrained(turkishModelPath)

# Sınıf ağırlıkları
total_samples = sum(class_counts.values())
class_weights = {label: total_samples / count for label, count in class_counts.items()}
class_weights_list = [class_weights[le.inverse_transform([i])[0]] for i in range(len(le.classes_))]
class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32)

loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

# Metric fonksiyonu
def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), dim=1)
    return {
        'accuracy': accuracy_score(p.label_ids, preds),
        'f1': f1_score(p.label_ids, preds, average='weighted')
    }

#Custom bir kayıp fonksiyonu kullanmak için Trainer sınıfını özelleştir
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# TrainingArguments
training_args = TrainingArguments(
    output_dir=f"{modelPath}/{modelFolder}",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_total_limit=1,
    logging_dir=f"{basePath}\\logs",
    logging_steps=10,
    warmup_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True
)

# Cross-Validation
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=30)

all_sentences = np.array(sentences)
all_labels_enc = np.array(labels_enc)

accuracy_scores = []
f1_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(all_sentences, all_labels_enc)):
    print(f"\n=== Fold {fold + 1}/{n_splits} ===")

    train_sentences, val_sentences = all_sentences[train_idx], all_sentences[val_idx]
    train_labels, val_labels = all_labels_enc[train_idx], all_labels_enc[val_idx]

    train_encodings = tokenizer(list(train_sentences), truncation=True, padding=True)
    val_encodings = tokenizer(list(val_sentences), truncation=True, padding=True)

    train_dataset = Dataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": train_labels
    })

    eval_dataset = Dataset.from_dict({
        "input_ids": val_encodings["input_ids"],
        "attention_mask": val_encodings["attention_mask"],
        "labels": val_labels
    })

    # model = BertForSequenceClassification.from_pretrained(turkishModelPath, num_labels=len(le.classes_))
    # Modeli path'ten yükle, yoksa yeni oluştur
    model_files = ["pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt.index", "flax_model.msgpack"]
    model_exists = any(os.path.exists(os.path.join(modelPath, f)) for f in model_files)

    if model_exists:
        model = BertForSequenceClassification.from_pretrained(modelPath)
    else:
        model = BertForSequenceClassification.from_pretrained(turkishModelPath, num_labels=len(le.classes_))

    # model.to("cuda" if torch.cuda.is_available() else "cpu")

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )

    trainer.train()

    predictions = trainer.predict(eval_dataset)
    preds = torch.argmax(torch.tensor(predictions.predictions), dim=1)

    acc = accuracy_score(eval_dataset["labels"], preds)
    f1 = f1_score(eval_dataset["labels"], preds, average='weighted')
    
    accuracy_scores.append(acc)
    f1_scores.append(f1)

    print(f"Fold {fold+1} Accuracy: {acc:.2%}")
    print(f"Fold {fold+1} F1 Score: {f1:.2%}")

    print(classification_report(eval_dataset["labels"], preds, target_names=le.classes_, zero_division=0))

    cm = confusion_matrix(eval_dataset["labels"], preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
    plt.title(f"Confusion Matrix - Fold {fold+1}")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(f"{resultSavingPath}/confusion_matrix_fold_{fold+1}.png")
    plt.close()

    torch.cuda.empty_cache()

print("\n=== Cross-Validation Sonuçları ===")
print(f"Ortalama Accuracy: {np.mean(accuracy_scores):.2%} (+/- {np.std(accuracy_scores):.2%})")
print(f"Ortalama F1 Score: {np.mean(f1_scores):.2%} (+/- {np.std(f1_scores):.2%})")

 # Model ve Etiketleri Kaydet
os.makedirs(f"{modelPath}/{modelFolder}", exist_ok=True)
tokenizer.save_pretrained(f"{modelPath}/{modelFolder}")
with open(f"{modelPath}/{modelFolder}/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

endTime = datetime.now()
totalSeconds = (endTime - startTime).total_seconds()
print(f"\nToplam süre: {totalSeconds:.2f} saniye ({totalSeconds / 60:.2f} dakika)")
