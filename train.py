import json
from preprocessing import tokenize, lemma, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

total_words = []
tags = []
word_pattern = []
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["symptom"]:
        w = tokenize(pattern)
        total_words.extend(w)
        word_pattern.append((w, tag))

ignore_word = ["?", "!", ".", ","]
total_words = [lemma(w) for w in total_words if w.text not in ignore_word]
total_words = sorted(set(total_words))
tags = sorted(set(tags))
print(tags)

X_train = []
y_train = []
for pattern_sentence, tag in word_pattern:
    bag = bag_of_words(pattern_sentence, total_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


batch_size = 8
input_size = len(X_train[0])
output_size = len(tags)
hidden_size = 8
learning_rate = 0.001
num_epochs = 1000


dataset = ChatDataset()

train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch+1}/{num_epochs},loss = {loss.item(): 4f}")

print(f"final loss, loss={loss.item():.4f}")


data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "total_words": total_words,
    "tags": tags,
}

FILE = "trained_data.pth"
torch.save(data, FILE)

print(f"training complete. file saved to {FILE}")
