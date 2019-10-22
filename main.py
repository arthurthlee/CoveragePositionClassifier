import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from string import punctuation
from collections import Counter

class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

        # define all layers


    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden


    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

if __name__ == '__main__':
    df = pd.DataFrame.from_csv('testdata.csv')
    rows = df.apply(lambda x: x.tolist(), axis=1)
    #print('Coverage: ' + rows.index.values)

    # Remove punctuation
    formatted_text = []
    for text in rows.values:
        if isinstance(text[0], str) == True:
            formatted_text.append(str(''.join([c for c in text[0] if c not in punctuation])))
        else:
            formatted_text.append('')
    #print(formatted_text)
    #print ('Number of coverage entries :', len(formatted_text))

    # Make giant list of strings with all words
    text_lists = []
    for text in formatted_text:
        text_lists.append(text)

    word_list = []
    for sentence in text_lists:
        text_in_sentence = sentence.split()
        for word in text_in_sentence:
            word_list.append(word)

    # Count all the words using Counter Method
    sorted_words = []
    count_words = Counter(word_list)
    total_words = len(word_list)
    sorted_words = count_words.most_common(total_words)

    print(sorted_words)

    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

    words_int = []
    for text in formatted_text:
        r = [vocab_to_int[w] for w in text.split()]
        words_int.append(r)
    print (words_int[0:3])

    encoded_labels = [1 if label =='t' else 0 for label in rows.index.values]
    print(encoded_labels)
    encoded_labels = np.array(encoded_labels)

    seq_length = 10

    features = np.zeros((len(words_int), seq_length), dtype = int)

    for i, review in enumerate(words_int):
        review_len = len(review)

        new = []
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = zeroes+review

        elif review_len > seq_length:
            new = review[0:seq_length]

        features[i,:] = np.array(new)

    print (features[:10,:])

    # Split the data into 80% training data, 10% test data and 10% validation data
    split_frac = 0.8
    train_x = features[0:int(split_frac*len(formatted_text))]
    train_y = encoded_labels[0:int(split_frac*len(formatted_text))]
    remaining_x = features[int(split_frac*len(formatted_text)):]
    remaining_y = encoded_labels[int(split_frac*len(formatted_text)):]
    valid_x = remaining_x[0:int(len(remaining_x)*0.5)]
    valid_y = remaining_y[0:int(len(remaining_y)*0.5)]
    test_x = remaining_x[int(len(remaining_x)*0.5):]
    test_y = remaining_y[int(len(remaining_y)*0.5):]

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    # dataloaders
    batch_size = 10
    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()
    print('Sample input size: ', sample_x.size()) # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size()) # batch_size
    print('Sample label: \n', sample_y)

    # Instantiate the model w/ hyperparams
    vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2

    #TODO: Implement below
    net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    print(net)

    # loss and optimization functions
    lr=0.001

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()

    # Get test data loss and accuracy

    test_losses = [] # track loss
    num_correct = 0

    # init hidden state
    h = net.init_hidden(batch_size)

    net.eval()
    # iterate over test data
    for inputs, labels in test_loader:

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # get predicted outputs
        output, h = net(inputs, h)

        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))


