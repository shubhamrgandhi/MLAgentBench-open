pip install sacremoses importlib_metadata

import datasets

# Load the IMDB dataset
raw_dataset = datasets.load_dataset("imdb")

# Inspect the dataset
print(raw_dataset)

from datasets import load_dataset
import torch
import pandas as pd

if __name__ == "__main__":
    imdb = load_dataset("imdb")

    # Define or load the model
    model = torch.hub.load('huggingface/pytorch-transformers', 'distilbert-base-uncased-finetuned-sst-2-english')

    #TODO: preprocess data

    #TODO: define model here
    # model = None

    #TODO: train model

    #evaluate model and print accuracy on test set, also save the predictions of probabilities per class to submission.csv
    submission = pd.DataFrame(columns=list(range(2)), index=range(len(imdb["test"])))
    acc = 0
    for idx, data in enumerate(imdb["test"]):
        text = data["text"]
        label = data["label"]
        pred = model(text) # TODO: replace with proper prediction
        pred = torch.softmax(pred, dim=0)
        submission.loc[idx] = pred.tolist()
        acc += int(torch.argmax(pred).item() == label)
    print("Accuracy: ", acc/len(imdb["test"]))
    
    submission.to_csv('submission.csv', index_label='idx')