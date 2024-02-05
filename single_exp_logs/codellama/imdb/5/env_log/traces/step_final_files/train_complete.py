from datasets import load_dataset
import torch
import pandas as pd

def load_data():
    return load_dataset("imdb")

def save_predictions(predictions, filename):
    submission = pd.DataFrame(columns=list(range(2)), index=range(len(predictions)))
    for i, pred in enumerate(predictions):
        submission.loc[i] = pred.tolist()
    submission.to_csv(filename, index_label='idx')

if __name__ == "__main__":
    imdb = load_data()

    #TODO: preprocess data

    #TODO: define model here
    model = None

    #TODO: train model

    #evaluate model and print accuracy on test set, also save the predictions of probabilities per class to submission.csv
    predictions = []
    for idx, data in enumerate(imdb["test"]):
        text = data["text"]
        label = data["label"]
        pred = model(text) # TODO: replace with proper prediction
        pred = torch.softmax(pred, dim=0)
        predictions.append(pred)
    save_predictions(predictions, 'submission.csv')