import nltk
import spacy

from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nlp = spacy.load("en_core_web_sm")
#nlp = spacy.load("tr_core_web_sm")

sid = SentimentIntensityAnalyzer() 

def Analyze_Sentiment(message):
    sentiment_scores = sid.polarity_scores(message)
    compound_score = sentiment_scores['compound']
    print(sentiment_scores)
    
    if compound_score >= 0.05:
        return "Olumlu"
    elif compound_score <= -0.05:
        return "Olumsuz"
    else:
        return "Nötr"

def recognize_intent(message):
    doc = nlp(message)
    print(doc)
    intent = None
    
    for token in doc:
        print(f"{token} : {token.pos_}")
        if token.pos_ == "VERB":
            intent = token.text
            break
    
    return intent if intent else "Bilinmeyen"

def extract_entities(message):
    doc = nlp(message)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.start_char, ent.end_char, ent.label_))
    
    return entities

def getMessageResponse(message):
    responses = {
    "hello": "Hi there! How can I help you today?",
    "name": "I'm a simple chatbot built with spaCy!",
    "weather": "I'm not sure, but I hope it's sunny where you are!",
    "bye": "Goodbye! Have a great day!",
}
    doc = nlp(message.lower())  # Kullanıcı mesajını analiz et
    for token in doc:
        print(f"lemma: {token.lemma_}")
        if token.lemma_ in responses:
            return responses[token.lemma_]
    return "I'm sorry, I didn't understand that."
    