import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Conv1D,\
    GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Dense
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

def lstm_bidirectional_v1(num_labels, max_len, max_words, embed_size, embedding_matrix):
    input = Input(shape=(max_len, ))
    x = Embedding(max_words, embed_size, weights=[embedding_matrix], trainable=False)(input)
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer="glorot_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    preds = Dense(num_labels, activation="sigmoid")(x)
    model = Model(input, preds)
    return model

def lstm_bidirectional_v2(num_labels, max_len, max_words, embed_size, embedding_matrix):
    input = Input(shape=(max_len, ))
    x = Embedding(max_words, embed_size, weights=[embedding_matrix], trainable=False)(input)
    x = Bidirectional(GRU(1024, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = Conv1D(512, kernel_size=3, padding='valid', kernel_initializer="glorot_uniform")(x)
    # x = Conv1D(512, kernel_size=3, padding='valid', kernel_initializer="glorot_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    preds = Dense(num_labels, activation="sigmoid")(x)
    model = Model(input, preds)
    return model