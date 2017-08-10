import numpy as np
import tensorflow as tf
import utils
import time

max_length = 20
input_seq_len = max_length
output_seq_len = max_length + 2
neuron_size = 512
learning_rate = 5e-3
batch_size = 64
training_epochs = 10
save_path = 'checkpoints/model'

def padding(x, y, length=20):
    from_sentence, to_sentence = [], []
    for i, _ in enumerate(x):
        if len(x[i]) <= length and len(y[i]) <= length:
            from_sentence.append(x[i] + (length - len(x[i])) * [from_word2idx['<pad>']])
            to_sentence.append([to_word2idx['<bos>']] + y[i] + [to_word2idx['<eos>']] + (length - len(y[i])) * [to_word2idx['<pad>']])

    return from_sentence, to_sentence

def get_batch(X, Y, i, batch_size=64):
    current_batch_x = X[i*batch_size:i*batch_size+batch_size]
    current_batch_y = Y[i*batch_size:i*batch_size+batch_size]
    return current_batch_x, current_batch_y

def sampled_loss(labels, logits):
    return tf.nn.sampled_softmax_loss(
                        weights = w_t,
                        biases = b,
                        labels = tf.reshape(labels, [-1, 1]),
                        inputs = logits,
                        num_sampled = 512,
                        num_classes = to_vocab_size)


def softmax(x):
    n = np.max(x)
    e_x = np.exp(x - n)
    return e_x / e_x.sum()

def feed_dict(x, y, batch_size = 64):
    feed = {}

    for i in range(input_seq_len):
        feed[encoder_inputs[i].name] = np.array([x[j][i] for j in range(len(x))], dtype=np.int32)

    for i in range(output_seq_len):
        feed[decoder_inputs[i].name] = np.array([y[j][i] for j in range(len(x))], dtype=np.int32)

    feed[targets[len(targets)-1].name] = np.full(shape=[batch_size], fill_value=to_word2idx['<pad>'], dtype=np.int32)

    for i in range(output_seq_len-1):
        batch_weights = np.ones(batch_size, dtype = np.float32)
        target = feed[decoder_inputs[i+1].name]
        for j in range(batch_size):
            if target[j] == to_word2idx['<pad>']:
                batch_weights[j] = 0.0
        feed[target_weights[i].name] = batch_weights

    feed[target_weights[output_seq_len-1].name] = np.zeros(batch_size, dtype=np.float32)

    return feed

def decode_output(output_seq):
    words = []
    for i in range(output_seq_len):
        smax = softmax(output_seq[i])
        idx = np.argmax(smax)
        words.append(to_idx2word[idx])
    return words


def backward_step(sess, feed):
    sess.run(optimizer, feed_dict = feed)


X, Y, from_word2idx, from_idx2word, to_word2idx, to_idx2word = utils.read_dataset('data/data.pkl')
X_train, Y_train = padding(X, Y, length=max_length)

from_vocab_size = len(from_word2idx)
to_vocab_size = len(to_word2idx)

encoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'encoder{}'.format(i)) for i in range(input_seq_len)]
decoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'decoder{}'.format(i)) for i in range(output_seq_len)]

targets = [decoder_inputs[i+1] for i in range(output_seq_len-1)]

targets.append(tf.placeholder(dtype = tf.int32, shape = [None], name = 'last_target'))
target_weights = [tf.placeholder(dtype = tf.float32, shape = [None], name = 'target_w{}'.format(i)) for i in range(output_seq_len)]

w_t = tf.get_variable('proj_w', [to_vocab_size, neuron_size], tf.float32)
b = tf.get_variable('proj_b', [to_vocab_size], tf.float32)
w = tf.transpose(w_t)
output_projection = (w, b)

outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                                            encoder_inputs,
                                            decoder_inputs,
                                            tf.contrib.rnn.BasicLSTMCell(neuron_size),
                                            num_encoder_symbols = from_vocab_size,
                                            num_decoder_symbols = to_vocab_size,
                                            embedding_size = 100,
                                            feed_previous = False,
                                            output_projection = output_projection,
                                            dtype = tf.float32)

loss_function = tf.contrib.legacy_seq2seq.sequence_loss(outputs, targets, target_weights, softmax_loss_function = sampled_loss)
outputs_proj = [tf.matmul(outputs[i], output_projection[0]) + output_projection[1] for i in range(output_seq_len)]
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)
init = tf.global_variables_initializer()
losses = []
saver = tf.train.Saver()

print '------------------TRAINING------------------'

with tf.Session() as sess:
    sess.run(init)
    loss  = np.inf
    t = time.time()
    # tf.train.Saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    for epoch in range(training_epochs):
        total_batch = int(len(X_train)/batch_size)        
        for i in range(total_batch):
            batch_x, batch_y = get_batch(X_train, Y_train, i, batch_size)
            feed = feed_dict(batch_x, batch_y, batch_size)
            backward_step(sess, feed)

            if i % 10 == 0:
                loss_value = sess.run(loss_function, feed_dict = feed)
                print 'batch : {}, loss: {}'.format(i,loss_value)

        loss_value = sess.run(loss_function, feed_dict = feed)
        print 'epoch: {}, loss: {}'.format(epoch, loss_value)
        losses.append(loss_value)
        if loss_value < loss :
            loss = loss_value
            saver.save(sess, save_path)
            print 'Checkpoint is saved'
            
            
    print 'Training time for {} steps: {}s'.format(steps, time.time() - t)



with tf.Graph().as_default():
    encoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'encoder{}'.format(i)) for i in range(input_seq_len)]
    decoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'decoder{}'.format(i)) for i in range(output_seq_len)]

    w_t = tf.get_variable('proj_w', [to_vocab_size, neuron_size], tf.float32)
    b = tf.get_variable('proj_b', [to_vocab_size], tf.float32)
    w = tf.transpose(w_t)
    output_projection = (w, b)
    
    outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                                                encoder_inputs,
                                                decoder_inputs,
                                                tf.contrib.rnn.BasicLSTMCell(neuron_size),
                                                num_encoder_symbols = from_vocab_size,
                                                num_decoder_symbols = to_vocab_size,
                                                embedding_size = 100,
                                                feed_previous = True, # <-----this is changed----->
                                                output_projection = output_projection,
                                                dtype = tf.float32)
    
    outputs_proj = [tf.matmul(outputs[i], output_projection[0]) + output_projection[1] for i in range(output_seq_len)]
    
    from_sentences = ["What' s your name", 'My name is', 'What are you doing', 'I am reading a book',\
                    'How are you', 'I am good', 'Do you speak English', 'What time is it', 'Hi', 'Goodbye', 'Yes', 'No']
    from_sentences_encoded = [[from_word2idx.get(word, 0) for word in from_sentence.split()] for from_sentence in from_sentences]
    
    for i in range(len(from_sentences_encoded)):
        from_sentences_encoded[i] += (max_length - len(from_sentences_encoded[i])) * [from_word2idx['<pad>']]
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:                
        saver.restore(sess, save_path)
        feed = {}
        for i in range(input_seq_len):
            feed[encoder_inputs[i].name] = np.array([from_sentences_encoded[j][i] for j in range(len(from_sentences_encoded))], dtype = np.int32)
            
        feed[decoder_inputs[0].name] = np.array([to_word2idx['<pad>']] * len(from_sentences_encoded), dtype = np.int32)
                
        output_sequences = sess.run(outputs_proj, feed_dict = feed)
        
        for i in range(len(from_sentences_encoded)):
            print '{}.\n--------------------------------'.format(i+1)
            ouput_seq = [output_sequences[j][i] for j in range(output_seq_len)]
            words = decode_output(ouput_seq)
        
            print from_sentences[i]
            for i in range(len(words)):
                if words[i] not in ['<pad>', '<bos>', '<eos>']:
                    print words[i],
            
            print '\n--------------------------------'