import pandas as pd
from string import punctuation
from collections import Counter

if __name__ == '__main__':
    df = pd.DataFrame.from_csv('testdata.csv')
    rows = df.apply(lambda x: x.tolist(), axis=1)
    print('Coverage: ' + rows.index.values)

    # Remove punctuation
    formatted_text = []
    for text in rows.values:
        if isinstance(text[0], str) == True:
            formatted_text.append(str(''.join([c for c in text[0] if c not in punctuation])))
        else:
            formatted_text.append('')
    #print(formatted_text)
    print ('Number of reviews :', len(formatted_text))

    # Make giant list of string with all words
    text_lists = []
    for text in formatted_text:
        text_lists.append(text)

    word_list = []
    for sentence in text_lists:
        text_in_sentence = sentence.split()
        for word in text_in_sentence:
            word_list.append(word)

    sorted_words = []
    # Count all the words using Counter Method
    count_words = Counter(word_list)
    total_words = len(word_list)
    sorted_words = count_words.most_common(total_words)

    print(sorted_words)