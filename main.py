import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from string import punctuation
from collections import Counter

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
    batch_size = 50
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
    net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    print(net)
    SentimentRNN(
        Embedding(74073, 400),
        LSTM(400, 256, num_layers=2, batch_first=True, dropout=0.5),
        Dropout(p=0.3),
        Linear(in_features=256, out_features=1, bias=True),
        Sigmoid()
    )
