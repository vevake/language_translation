import cPickle as pickle
import os
from collections import Counter

def read_file(file_to_read):
    sentences = []
    with open(file_to_read, 'r') as f:
        for sentence in f:
            sentence_stripped = sentence.strip()
            sentences.append(sentence_stripped)
    return sentences

def create_dataset(en_file, de_file):
    en_sentences = read_file(en_file)
    de_sentences = read_file(de_file)

    en_vocab = Counter(word.strip(',." ;:)(][?!') for sentence in en_sentences for word in sentence.split())
    de_vocab = Counter(word.strip(',." ;:)(][?!') for sentence in de_sentences for word in sentence.split())

    en_vocab = map(lambda x: x[0], sorted(en_vocab.items(), key = lambda x: -x[1]))
    de_vocab = map(lambda x: x[0], sorted(de_vocab.items(), key = lambda x: -x[1]))

    start_idx = 2
    en_word2idx = dict([(word, idx+start_idx) for idx, word in enumerate(en_vocab)])
    en_word2idx['<unk>'] = 0
    en_word2idx['<pad>'] = 1

    en_idx2word = dict([(idx, word) for word, idx in en_word2idx.iteritems()])

    de_word2idx = dict([(word, idx+start_idx) for idx, word in enumerate(de_vocab)])
    de_word2idx['<unk>'] = 0
    de_word2idx['<pad>'] = 1
    de_idx2word = dict([(idx, word) for word, idx in de_word2idx.iteritems()])

    X = [[en_word2idx.get(word.strip(',." ;:)(][?!'), 0) for word in sentence.strip()] for sentence in en_sentences]
    Y = [[de_word2idx.get(word.strip(',." ;:)(][?!'), 0) for word in sentence.strip()] for sentence in de_sentences]

    return X, Y, en_word2idx, en_idx2word, en_vocab, de_word2idx, de_idx2word, de_vocab

def save_dataset(file_path_to_save, obj):
    with open(file_path_to_save, 'wb') as f:
        pickle.dump(obj, f, -1)
    print 'Dataset saved successfully'

def main():
    current_directory = os.getcwd()
    en_file = os.path.join(current_directory, 'data/OpenSubtitles.de-en.en')
    de_file = os.path.join(current_directory, 'data/OpenSubtitles.de-en.de')
    file_to_save_dataset = os.path.join(current_directory, 'data/data.pkl')

    save_dataset(file_to_save_dataset, create_dataset(en_file, de_file))

if __name__ =='__main__':
    main()