"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]


#%%
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

scenes = text.split('\n\n')
sentence_count_scene = [scene.count('\n') for scene in scenes]
sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
word_count_sentence = [len(sentence.split()) for sentence in sentences]


print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
print('Number of scenes: {}'.format(len(scenes)))
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

print('Number of lines: {}'.format(len(sentences)))
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


#%% Implement Preprocessing Functions
# Lookup Table

import numpy as np
import problem_unittests as tests

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
      
    # Create your dictionary that maps vocab words to integers here
    from collections import Counter
    counts = Counter(text)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab)}

    # Convert the integers into words
    int_to_vocab = { int_var:word for word, int_var in vocab_to_int.items() }
    assert len(int_to_vocab) == len(vocab_to_int)
    return vocab_to_int, int_to_vocab


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)

#%% Tokenize Punctuation

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    dict_token = {  '.'  : '||Period||',
                    ','  : '||Comma||',
                    '"'  : '||Quotation||',
                    ';'  : '||Semicolon||',
                    '!'  : '||Exclamation||',
                    '?'  : '||Question||',
                    '('  : '||Left_Parentheses||',
                    ')'  : '||Right_Parentheses||',
                    '--' : '||Dash||',
                    '\n' : '||Return||'}
    
    return dict_token

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)

#%% Preprocess all the data and save it

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

#%% Check Point
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

#%% Build the Neural Network
# Check the Version of TensorFlow and Access to GPU
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

#%% Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


#%% Input
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    input = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'targets')
    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
    
    return input, targets, learning_rate

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)

#%% Build RNN Cell and Initialize
def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """    
    lstm_layers = 1 
    
    # Your basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    # Add dropout to the cell
    # lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([lstm] * lstm_layers)
      
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity( initial_state, name = 'initial_state')
    
    return cell, initial_state

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)

#%% Word Embedding
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable( tf.random_normal((vocab_size, embed_dim)) )
    embed = tf.nn.embedding_lookup(embedding, input_data)
    
    return embed

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)


#%% Build RNN
def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)   

    final_state = tf.identity(final_state, name = 'final_state')
    
    return outputs, final_state

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)


#%% Build the Neural Network
def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """    
    inputs = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, inputs)
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn = None)
    
    return logits, final_state

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)

#%% Batches
def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    
    n_batches = len(int_text)//(batch_size*seq_length)
    
    int_text = int_text[:n_batches*batch_size*seq_length]
    x = int_text
    y = x[1:] + [x[0]]
    # print('len(x)',len(x))
    # print('batch_size',batch_size)
    # print('seq_length',seq_length)
    # print('n_batches',n_batches)
    # print('n_batches*batch_size*seq_length', n_batches*batch_size*seq_length)
    # print('len(x)',len(x))
    
    batches = np.zeros((n_batches, 2 ,batch_size, seq_length), dtype = np.int32)
    # print('Shape before fill', np.array(batches).shape)
    
    ii = 0
    i_seq = 0
        
    while i_seq<batch_size:
        # Loop the batches adding a new sample
        for i_batch, batch in enumerate(batches):
        
            new_input  = np.array(x[ii:ii+seq_length])
            new_target = np.array(y[ii:ii+seq_length])
            # new_input  = 'batch('+str(i_batch)+'),input' 
            # new_target = 'batch('+str(i_batch)+'),output'
            # print('batch', i_batch,', i_seq',i_seq,' - new_input', new_input)
            batches[i_batch][0][i_seq] = new_input
            batches[i_batch][1][i_seq] = new_target
            
            ii += seq_length
            
        i_seq += 1
        
            
        
    # print('BEFORE converting to np --------------------------------------------------')
    # print(batches)
    # print('batches[0] (first batch)')
    # print(batches[0])
    # print('batches[0] (inputs)')
    # print(batches[0][0])
    # print('AFTER converting to np --------------------------------------------------')
    # batches = np.array(batches)
    # print(batches)
    # print('batches[0]')
    # print(batches[0])
    # print('batches[0] (inputs)')
    # print(batches[0][0])
    # print('                       --------------------------------------------------')
    
    # print('Should be: (number of batches, 2, batch size, sequence length) = ({},2,{},{})'.format(n_batches, batch_size,seq_length) )
    # print('But it is: ',batches.shape)
    
    
    # n_batches = len(x)//batch_size
    # x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    # for ii in range(0, len(x), batch_size):
        # yield x[ii:ii+batch_size], y[ii:ii+batch_size]
    
    return batches

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
batches = get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2)
# print('batches')
# print(batches)
# print('batches[0]')
# print(batches[0])
tests.test_get_batches(get_batches)


#%%
# Number of Epochs
num_epochs = 300
# Batch Size
batch_size = 256
# RNN Size
rnn_size = 256
# Embedding Dimension Size
embed_dim = 300
# Sequence Length
seq_length = 50
# Learning Rate
learning_rate = 5e-3 
# Show stats for every n number of batches
show_every_n_batches = 200

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""

from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

    
# Train
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
    
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))

