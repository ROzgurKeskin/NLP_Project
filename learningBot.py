from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle
import json
import random
from datetime import datetime

# === Yollar ===
#modelPath="C:\\PythonProject\\Github Project\\NLP_Project\\model" 
modelPath="C:\\PythonProject\\Github Project\\NLP_Project\\model"
intentPath="C:\\PythonProject\\Github Project\\NLP_Project\\intents.json"
lowConfidenseLog="C:\\PythonProject\\Github Project\\NLP_Project\\low_confidence_logs.txt"

# === Model, Tokenizer, Etiketler ===
model = BertForSequenceClassification.from_pretrained(modelPath, ignore_mismatched_sizes=True)
tokenizer = BertTokenizer.from_pretrained(modelPath)

with open(f"{modelPath}/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open(intentPath, "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

# Tahmin güvenilirlik değeri
def softmax_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=1)

# === Yanıt Fonksiyonu ===
def get_response(text, threshold=0.88, log_file=lowConfidenseLog):
    # Girdi vektörleştirme
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    
    # Tahmin edilen intent ve olasılık
    probabilities = softmax_with_temperature(outputs.logits, temperature=0.8)
    
    max_prob, predicted = torch.max(probabilities, dim=1)
    max_prob = max_prob.item()
    predicted = predicted.item()
    
    tag = le.inverse_transform([predicted])[0]

    print(f"[DEBUG] Tahmin edilen intent: {tag}, Olasılık: {max_prob:.2f}")

    # Eğer olasılık eşikten düşükse, logla
    if max_prob < threshold:
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"Low confidence prediction: Text='{text}', Tag='{tag}', Probability={max_prob:.2f}\n")
        return {"ResponseText": "Ne demek istediğini anlayamadım.", "Tag": "Bilinmiyor"}

    for intent in intents:
        if intent["tag"] == tag:
            responseText = random.choice(intent["responses"])

            # BONUS: Saat intent'i için dinamik sistem saati ekle
            if tag == "saat":
                responseText = f"{responseText} Şu an: {datetime.now().strftime('%H:%M')}"
            
            # BONUS: Tarih intent'i için dinamik tarih ekle
            if tag == "tarih":
                responseText = f"{responseText} Bugün: {datetime.now().strftime('%d.%m.%Y')}"

            return {"ResponseText": responseText, "Tag": tag}

    return {"ResponseText": "Ne demek istediğini anlayamadım.", "Tag": "Bilinmiyor"}
    
