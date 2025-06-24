import json
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer,XLMRobertaForSequenceClassification, BertForSequenceClassification, Trainer, TrainingArguments,XLMRobertaTokenizer 
import torch
from datasets import Dataset
startTime = datetime.now()
# === Ayarlar === 
basePath="C:\\PythonProject\\Github Project\\NLP_Project"
modelPath="C:\\PythonProject\\Github Project\\NLP_Project\\model\\fine_tuned"
intentPath="C:\\PythonProject\\Github Project\\NLP_Project\\intents.json"
turkishModelPath="dbmdz/bert-base-turkish-cased";
#turkishModelPath="xlm-roberta-base";

# === Veriyi Yükle ===
with open(intentPath, "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = []
labels = []

# ===Sadece belirlenen sınıflarını filtrele ===
selected_tags = []
for intent in data["intents"]:
    if len(selected_tags)==0 or (len(selected_tags)>0 and (intent["tag"] in selected_tags)):
        for pattern in intent["patterns"]:
            sentences.append(pattern)
            labels.append(intent["tag"])

# Sınıf dağılımını kontrol et
from collections import Counter
class_counts = Counter(labels)
print("Sınıf Dağılımı:", class_counts)

#Sınıf dağılımını görselleştir
plt.figure(figsize=(10, 6))

# === Label Encoding ===
# try to load existing label encoder
try:
    with open(f"{modelPath}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
except FileNotFoundError:
    le = LabelEncoder()
    
# Plot the class distribution
plt.bar(class_counts.keys(), class_counts.values())
labels_enc = le.fit_transform(labels)

# le = LabelEncoder()
# labels_enc = le.fit_transform(labels)

# === Tokenizer ve Dataset ===
tokenizer_path = f"{modelPath}"
if os.path.exists(tokenizer_path) and os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
else:
    tokenizer = BertTokenizer.from_pretrained(turkishModelPath)

# === Model Oluştur ===
# model = BertForSequenceClassification.from_pretrained(modelPath)
# Modeli path'ten yükle, yoksa yeni oluştur

try:
    model = BertForSequenceClassification.from_pretrained(modelPath)
except Exception as e:
    print(f"Model yüklenirken hata oluştu, yeni oluşturuldu: {e}")
    model = BertForSequenceClassification.from_pretrained(turkishModelPath, num_labels=len(le.classes_))


#Tokenizer'ı kaydet ve tekrar kullan
tokenizer_path = f"{modelPath}"
if not os.path.exists(tokenizer_path):
    tokenizer.save_pretrained(tokenizer_path)

# Eğitim ve doğrulama setlerini ayır
from sklearn.model_selection import train_test_split

# Stratified split
train_sentences, eval_sentences, train_labels, eval_labels = train_test_split(
    sentences, labels_enc, test_size=0.2, stratify=labels_enc, random_state=42
)

# Tokenize edilmiş veriyi yeniden oluştur
train_encodings = tokenizer(train_sentences, truncation=True, padding=True)
eval_encodings = tokenizer(eval_sentences, truncation=True, padding=True)

train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})

eval_dataset = Dataset.from_dict({
    "input_ids": eval_encodings["input_ids"],
    "attention_mask": eval_encodings["attention_mask"],
    "labels": eval_labels
})


# Sınıf ağırlıklarını hesapla
total_samples = sum(class_counts.values())
class_weights = {label: total_samples / count for label, count in class_counts.items()}

# Ağırlıkları sıralı bir listeye dönüştür
class_weights_list = [class_weights[le.inverse_transform([i])[0]] for i in range(len(le.classes_))]

# Torch tensora dönüştür
class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32).to(model.device)

from torch.nn import CrossEntropyLoss

# Özel bir kayıp fonksiyonu tanımla
loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

from transformers import get_scheduler

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=len(train_dataset))

from sklearn.metrics import f1_score
from transformers import EarlyStoppingCallback

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
        loss = loss_fn(outputs.logits, labels)  # Özel kayıp fonksiyonu kullanımı
        return (loss, outputs) if return_outputs else loss

# === Eğitim Ayarları ===
training_args = TrainingArguments(
    #output_dir=modelPath,
    output_dir=f"{modelPath}",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_total_limit=1,
    #weight_decay=0.01,  # Ağırlık çürümesi
    logging_dir=f"{basePath}\\logs",
    logging_steps=10,
    warmup_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model='f1',  # veya 'accuracy'
    greater_is_better=True
)

# CustomTrainer ile Trainer başlat
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]  # X epoch boyunca gelişme olmazsa dur
)

# === Eğitimi Başlat ===
trainer.train()

# === Test Verisinde Değerlendir ===
predictions = trainer.predict(eval_dataset)
preds = torch.argmax(torch.tensor(predictions.predictions), dim=1)

# === Doğruluk Hesapla ===
acc = accuracy_score(eval_dataset["labels"], preds)
acc2 = f1_score(eval_dataset["labels"], preds, average='weighted') 

print(f"\n Accuracy (Doğruluk): {acc:.2%}")
print(f"\n F1 Score : {acc2:.2%}")

from sklearn.metrics import classification_report
print(classification_report(eval_dataset["labels"], preds, target_names=le.classes_, zero_division=0))

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

# Model ve Etiketleri Kaydet
os.makedirs(f"{modelPath}/fine_tuned", exist_ok=True)
model.save_pretrained(f"{modelPath}/fine_tuned")
tokenizer.save_pretrained(f"{modelPath}/fine_tuned")
with open(f"{modelPath}/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model başarıyla kaydedildi ve yüklendi.")
endTime = datetime.now()
totalSeconds= (endTime-startTime).total_seconds()
# toplam dakikayı hesapla
totalMinutes=totalSeconds/60
print(f"öğrenme süresi:{totalSeconds} saniye")
print(f"öğrenme süresi:{totalMinutes} dakika")