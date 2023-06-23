from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

chat = "@rogers aloo today we are cebrating the entire period"

chat_words = []

for word in chat.split(" "):
    if word.startswith("@") and len(word) >1:
        word = "@user"
        
    elif word.startswith("http"):
        word = "http"
    chat_words.append(word)



print(chat_words)