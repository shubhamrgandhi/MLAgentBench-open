import datasets
import torch
import pandas as pd

# Load the dataset
dataset = datasets.load_dataset("imdb")

if __name__ == "__main__":
    imdb = dataset

    # Preprocess the data
    import string

    def preprocess_text(text):
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        return text

    dataset = dataset.map(lambda x: {'text': preprocess_text(x['text'])}, batched=True)

    #TODO: define model here
    model = None

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