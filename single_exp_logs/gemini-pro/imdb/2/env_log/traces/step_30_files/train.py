from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

from datasets import load_dataset
import torch
import pandas as pd

if __name__ == "__main__":
    imdb = load_dataset("imdb")

    # preprocess data
    max_length = 512
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    for idx, data in enumerate(imdb["train"]):
        text = data["text"]
        truncated_text = tokenizer.truncate_sequences(text, max_length=max_length)
        imdb["train"][idx]["text"] = truncated_text
    for idx, data in enumerate(imdb["test"]):
        text = data["text"]
        truncated_text = tokenizer.truncate_sequences(text, max_length=max_length)
        imdb["test"][idx]["text"] = truncated_text

    # define model here
    #model = None

    # train model

    # evaluate model and print accuracy on test set, also save the predictions of probabilities per class to submission.csv
    submission = pd.DataFrame(columns=list(range(2)), index=range(len(imdb["test"])))
    acc = 0
    for idx, data in enumerate(imdb["test"]):
        text = data["text"]
        label = data["label"]
        inputs = tokenizer(text, return_tensors="pt")
        pred = model(**inputs) # TODO: replace with proper prediction
        pred = torch.softmax(pred.logits, dim=0)
        submission.loc[idx] = pred.tolist()
        acc += int(torch.argmax(pred).item() == label)
    print("Accuracy: ", acc/len(imdb["test"]))
    
    submission.to_csv('submission.csv', index_label='idx')