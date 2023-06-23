from transformers import AutoTokenizer, \
    AutoModelForSequenceClassification
from scipy.special import softmax

chat = "@rogers aloo today we are cebrating the entire period"


def split_chat(agent_chat:str):
    """Split the chats recived from the customer by the agent

    Args:
        agent_chat (str): chat from the customer

    Returns:
        _type_: split and joined words from the chat
    """
    chat_words = []

    for word in agent_chat.split(" "):
        if word.startswith("@") and len(word) >1:
            word = "@user"
            
        elif word.startswith("http"):
            word = "http"
        else:
            chat_words.append(word)

    return " ".join(chat_words)


# load the model to use for the sentiment anlysis
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ["Negative", "Neutral", "Positive"]


def sentiment_anlysis(joined_chat:str):
    """Performs a sentimental anlysis on the chats

    Args:
        joined_chat (str): string of the split and joined chat from customer

    Returns:
        _type_: split and joined words from the cha
    """
    encoded_chat = tokenizer(joined_chat, return_tensors= 'pt')
    output = model(encoded_chat[''])

def main():
    joined = split_chat(chat)
    sentiment_anlysis(joined)


if __name__ == '__main__':
    main()

