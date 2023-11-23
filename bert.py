from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
tokens = tokenizer.encode('It was good but couldve been better. Great', return_tensors='pt')

if __name__ == "__main__":
    result = model(tokens)
    print(result)
    result.logits
    int(torch.argmax(result.logits))+1