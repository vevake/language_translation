import cPickle as pickle
import os
from collections import Counter

def read_file(file_to_read):
    '''
    This function reads the input file and outputs the sentences in the file as a list of list
    '''
    sentences = []
    with open(file_to_read, 'r') as f:
        for sentence in f:
            sentence_stripped = sentence.strip()
            sentences.append(sentence_stripped)
    return sentences

def create_dataset(from_file, to_file):
    from_sentences = read_file(from_file)
    to_sentences = read_file(to_file)

    from_vocab = Counter(word.strip(',." ;:)(][?!') for sentence in from_sentences for word in sentence.split())
    to_vocab = Counter(word.strip(',." ;:)(][?!') for sentence in to_sentences for word in sentence.split())

    from_vocab = map(lambda x: x[0], sorted(from_vocab.items(), key=lambda x: -x[1]))
    to_vocab = map(lambda x: x[0], sorted(to_vocab.items(), key=lambda x: -x[1]))

    from_vocab = from_vocab[:int(len(from_vocab)*0.9)]
    to_vocab = to_vocab[:int(len(to_vocab)*0.9)]

    start_idx = 2
    from_word2idx = dict([(word, idx+start_idx) for idx, word in enumerate(from_vocab)])
    from_word2idx['<unk>'] = 0
    from_word2idx['<pad>'] = 1

    from_idx2word = dict([(idx, word) for word, idx in from_word2idx.iteritems()])

    start_idx = 4
    to_word2idx = dict([(word, idx+start_idx) for idx, word in enumerate(to_vocab)])
    to_word2idx['<unk>'] = 0
    to_word2idx['<pad>'] = 1
    to_word2idx['<bos>'] = 2
    to_word2idx['<eos>'] = 3
    to_idx2word = dict([(idx, word) for word, idx in to_word2idx.iteritems()])

    x = [[from_word2idx.get(word.strip(',." ;:)(][?!'), 0) for word in sentence.split()] for sentence in from_sentences]
    y = [[to_word2idx.get(word.strip(',." ;:)(][?!'), 0) for word in sentence.split()] for sentence in to_sentences]

    X, Y = [], []

    for i, _ in enumerate(x):
        if abs(len(x[i]) - len(y[i])) <= 0.3 * max(len(x[i]), len(y[i])):
            X.append(x[i])
            Y.append(y[i])

    return X, Y, from_word2idx, from_idx2word, to_word2idx, to_idx2word


def save_dataset(file_path_to_save, obj):
    with open(file_path_to_save, 'wb') as doc:
        pickle.dump(obj, doc, -1)
    print 'Dataset saved successfully'

def read_dataset(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def main():
    current_directory = os.getcwd()
    from_file = os.path.join(current_directory, 'data/OpenSubtitles.de-en.en')
    to_file = os.path.join(current_directory, 'data/OpenSubtitles.de-en.de')
    file_to_save_dataset = os.path.join(current_directory, 'data/data.pkl')

    save_dataset(file_to_save_dataset, create_dataset(from_file, to_file))

if __name__ == '__main__':
    main()
