""""
John Jefferson III and Michael Patel
June 2019
Python 3.6.5
TF 1.12.0

Project Description:
    - Generate our own Taylor Swift music lyrics using RNNs!

Datasets:
    csv file of Taylor Swift lyrics

Notes:

Things to examine:

"""
################################################################################
# Imports
import os
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow as tf


################################################################################
# Model hyperparameters
MAX_SEQ_LENGTH = 50
BUFFER_SIZE = 10000
BATCH_SIZE = 64
EMBEDDING_DIM = 512
NUM_RNN_UNITS = 2048
NUM_EPOCHS = 30
CHECKPOINT_PERIOD = NUM_EPOCHS  # how frequently to save checkpoints

# Generation parameters
START_STRING = "And "
NUM_CHAR_GEN = 300  # number of generated characters
TEMPERATURE = 0.5


################################################################################
# Data Preprocessing
class Dataset:
    def __init__(self):
        self.encoding = "windows-1252"

        # csv dataset file
        input_csv = "taylor_swift_lyrics.csv"
        column_header = "lyric"
        input_file = os.path.join(os.getcwd(), "data\\" + input_csv)

        # lyrics df
        self.lyrics_df = pd.read_csv(input_file, usecols=[column_header], encoding=self.encoding)


################################################################################
# RNN
def build_model(vocab_size, embedding_dim, num_rnn_units, batch_size):
    model = tf.keras.Sequential()

    # Embedding layer
    model.add(tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        batch_size=batch_size
    ))

    # GRU layers
    if tf.test.is_gpu_available():
        model.add(tf.keras.layers.CuDNNGRU(
            units=num_rnn_units,
            return_sequences=True,
            stateful=True
        ))

        model.add(tf.keras.layers.CuDNNGRU(
            units=num_rnn_units,
            return_sequences=True,
            stateful=True
        ))

    else:
        model.add(tf.keras.layers.GRU(
            units=num_rnn_units,
            return_sequences=True,
            stateful=True
        ))

        model.add(tf.keras.layers.GRU(
            units=num_rnn_units,
            return_sequences=True,
            stateful=True
        ))

    # Fully Connected layer
    model.add(tf.keras.layers.Dense(
        units=vocab_size
    ))

    return model


################################################################################
# Loss function
def loss_fn(labels, logits):
    return tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )


# Callbacks
def build_callbacks(chkpt_dir):
    history_file = os.path.join(chkpt_dir, "checkpoint_{epoch}")

    # save callback
    sc = tf.keras.callbacks.ModelCheckpoint(
        filepath=history_file,
        save_weights_only=True,
        period=CHECKPOINT_PERIOD,
        verbose=1
    )

    # TensorBoard callback
    tb = tf.keras.callbacks.TensorBoard(log_dir=chkpt_dir)

    return sc, tb


# Generate output
def generate(model, start_string):
    input_eval = [char2idx[c] for c in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    gen_text = []

    model.reset_states()

    for i in range(NUM_CHAR_GEN):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions /= TEMPERATURE

        id_predictions = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([id_predictions], 0)

        gen_text.append(idx2char[id_predictions])

    return start_string + "".join(gen_text)


################################################################################
# Main
if __name__ == "__main__":
    # enable eager execution
    tf.enable_eager_execution()

    # print out TF version
    print("TF version: {}".format(tf.__version__))

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    d = Dataset()
    df = d.lyrics_df

    # build list of lyrics from df
    lyrics_list = ["".join(i) for i in df.values]

    # convert list into one string (char list)
    lyrics = "\n".join(lyrics_list)
    # print(lyrics)
    # print("Length of lyrics text: {}".format(len(lyrics)))

    # Tokenization: string => char tokens
    unique_chars = sorted(set(lyrics))
    vocab_size = len(unique_chars)
    # print("Unique characters: {}".format(unique_chars))
    print("Number of unique characters: {}".format(vocab_size))

    # Vectorization: convert char to vectors of int
    # create mapping from char to int
    char2idx = {u: i for i, u in enumerate(unique_chars)}

    # create mapping from int to char
    idx2char = {i: u for i, u in enumerate(unique_chars)}

    # create input and target sequences
    input_seqs = []
    target_seqs = []

    # build list of sequences of indices
    for i in range(0, len(lyrics)-MAX_SEQ_LENGTH, MAX_SEQ_LENGTH):
        # create batches of char
        input_chars = lyrics[i: i+MAX_SEQ_LENGTH]
        target_chars = lyrics[i+1: i+1+MAX_SEQ_LENGTH]

        # convert each char in batch to int
        input_seqs.append([char2idx[i] for i in input_chars])
        target_seqs.append([char2idx[t] for t in target_chars])

    # shape: (n, MAX_SEQ_LENGTH) where n is the number of index sequences
    print("Shape of input sequences: {}".format(str(np.array(input_seqs).shape)))
    print("Shape of target sequences: {}".format(str(np.array(target_seqs).shape)))

    # use tf.data.Dataset to create batches and shuffle => TF model
    # (features, labels) == (input_seqs, target_seqs)
    sequences = tf.data.Dataset.from_tensor_slices((input_seqs, target_seqs))
    sequences = sequences.shuffle(buffer_size=BUFFER_SIZE)
    sequences = sequences.batch(batch_size=BATCH_SIZE, drop_remainder=True)

    print("Shape of batches: {}".format(sequences))

    # ----- MODEL ----- #
    m = build_model(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_rnn_units=NUM_RNN_UNITS,
        batch_size=BATCH_SIZE
    )

    m.summary()

    # loss function and optimization
    m.compile(
        loss=loss_fn,
        optimizer=tf.train.AdamOptimizer()
    )

    # callbacks for checkpoints, Tensorboard
    dir_name = "Results\\" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    checkpoint_dir = os.path.join(os.getcwd(), dir_name)
    save_callback, tb_callback = build_callbacks(checkpoint_dir)

    # train model
    history = m.fit(
        x=sequences.repeat(),
        epochs=NUM_EPOCHS,
        callbacks=[save_callback, tb_callback],
        steps_per_epoch=len(lyrics)//MAX_SEQ_LENGTH//BATCH_SIZE,
        verbose=1
    )

    # ----- GENERATE ----- #
    # run model with different batch size, so need to rebuild model
    m = build_model(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_rnn_units=NUM_RNN_UNITS,
        batch_size=1  # seed the model
    )

    m.load_weights(tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir))
    m.build(tf.TensorShape([1, None]))
    m.summary()

    generated = generate(model=m, start_string=START_STRING)

    # write generated output to text file
    print("\nWriting generated output to text file...")
    output_file = os.path.join(checkpoint_dir, "output.txt")

    with open(output_file, "w+") as f:
        # write hyperparameters
        f.write("Number of Epochs: {}".format(NUM_EPOCHS))
        f.write("\nBatch Size: {}".format(BATCH_SIZE))
        f.write("\nMaximum Sequence Length: {}".format(MAX_SEQ_LENGTH))
        f.write("\nEmbedding Dimension: {}".format(EMBEDDING_DIM))
        f.write("\nNumber of RNN Units: {}".format(NUM_RNN_UNITS))

        f.write("\nNumber of Characters Generated: {}".format(NUM_CHAR_GEN))
        f.write("\nTemperature: {}".format(TEMPERATURE))

        # write generated output
        f.write("\n\n################################################################################")
        f.write("\nGENERATED OUTPUT:")
        f.write("\n" + generated)
        f.write("\n################################################################################")
