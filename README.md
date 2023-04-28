# symptom-diagnosis-chatbot
A chatbot trained on json file consisting of symptoms and their respective diagnosis.

# Preprocessing:
Preprocessing was done using spacy which involves tokenization of text, and further removal of stop words for efficient training,then lemmatization of each word to get its base form.
The words are then returned as  a binary bag of words vector representation of the sentence as a numpy array.

#Training:
The model was implemented using PyTorch , the implementation is straightforward with a Feed Forward Neural net with 2 hidden layers.
The model is a three-layer feedforward neural network with ReLU activation function.
The PyTorch criterion and optimizer were set up using the Cross Entropy Loss and the Adam optimizer, respectively.
The model is trained for 1000 epochs, and during each epoch. The model predictions are compared with the actual labels using the Cross Entropy Loss function, and the optimizer is used to update the model parameters to minimize the loss.

The bag of words vector is then passed through the trained neural network to obtain a prediction of the intent of the input message. The highest-scoring output from the neural network is selected as the predicted intent.

If the predicted intent has a probability above a threshold of 0.75, the chatbot selects a response at random from the list of responses associated with the predicted intent. Otherwise, the chatbot responds with a default message "I do not understand...".
