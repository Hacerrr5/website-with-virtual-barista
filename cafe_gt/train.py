# -----------------------------------------------
# train.py
# -----------------------------------------------
# This script trains the chatbot model (virtual barista)
# using PyTorch. It handles data preprocessing, model 
# training, and saving the trained model.
# -----------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import json
import os

# nltk.download('punkt') # Uncomment this line if running for the first time

# -----------------------------------------------
# Utility functions for preprocessing
# -----------------------------------------------
stemmer = PorterStemmer()

def tokenize(sentence):
    """Split sentence into words (tokens)."""
    return nltk.word_tokenize(sentence)

def stem(word):
    """Stem a word to its root form."""
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    Create a bag-of-words array:
    1 for each known word that exists in the sentence, 0 otherwise.
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# -----------------------------------------------
# Define the neural network architecture
# -----------------------------------------------
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# -----------------------------------------------
# Training function
# -----------------------------------------------
def train_model():
    # Load the intents file
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    # Process patterns and tags
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    # Remove punctuation and stem
    ignore_words = ['?', '!', '.', ',']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(list(set(all_words)))
    tags = sorted(list(set(tags)))

    # Create training data
    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # -----------------------------------------------
    # Custom dataset for PyTorch
    # -----------------------------------------------
    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train
        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]
        def __len__(self):
            return self.n_samples

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0)

    # -----------------------------------------------
    # Initialize model, loss, and optimizer
    # -----------------------------------------------
    input_size = len(all_words)
    hidden_size = 8
    output_size = len(tags)
    
    model = NeuralNet(input_size, hidden_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # -----------------------------------------------
    # Train the model
    # -----------------------------------------------
    num_epochs = 500
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            outputs = model(words)
            loss = criterion(outputs, labels.long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Training completed. Final Loss: {loss.item():.4f}')

    # -----------------------------------------------
    # Save the trained model
    # -----------------------------------------------
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }
    
    # Save directory
    output_dir = r"C:\Users\Casper\Desktop\kahve"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, 'chatbot_model.pth')
    torch.save(data, file_path)

    print(f'Model saved at: {file_path}')

# -----------------------------------------------
# Run training
# -----------------------------------------------
if __name__ == '__main__':
    train_model()
