from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle
import json
import random
from datetime import datetime

# === Yollar ===
modelPath="C:\\PythonProject\\Github Project\\UskudarProject\\model" 
intentPath="C:\\PythonProject\\Github Project\\UskudarProject\\intents.json"

# === Model, Tokenizer, Etiketler ===
model = BertForSequenceClassification.from_pretrained(modelPath, ignore_mismatched_sizes=True)
tokenizer = BertTokenizer.from_pretrained(modelPath)

with open(f"{modelPath}/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open(intentPath, "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

# === Yanıt Fonksiyonu ===
def get_response(text):
    # Girdi vektörleştirme
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    
    # Tahmin edilen intent
    predicted = torch.argmax(outputs.logits, dim=1).item()
    tag = le.inverse_transform([predicted])[0]

    print(f"[DEBUG] Tahmin edilen intent: {tag}")

    for intent in intents:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])

            # BONUS: Saat intent'i için dinamik sistem saati ekle
            if tag == "saat":
                return f"{response} Şu an: {datetime.now().strftime('%H:%M')}"
            
            # BONUS: Tarih intent'i için dinamik tarih ekle
            if tag == "tarih":
                return f"{response} Bugün: {datetime.now().strftime('%d.%m.%Y')}"

            return response

    return "Ne demek istediğini anlayamadım."

# === Ana Döngü ===
print("ChatBot: Merhaba! (Çıkmak için 'çık' yaz.)")
while True:
    text = input("Sen: ")
    if text.lower() == "çık":
        print("ChatBot: Görüşmek üzere!")
        break
    print("ChatBot:", get_response(text))
