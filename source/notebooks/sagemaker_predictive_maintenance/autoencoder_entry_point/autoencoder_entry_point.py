# Autoencoder based on: https://towardsdatascience.com/predictive-maintenance-of-turbofan-engine-64911e39c367

import argparse
import pandas as pd
import numpy as np
import itertools
import logging
import random
import os

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *

def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    return logger


def parse_args():
    parser = argparse.ArgumentParser()

    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--training_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--num_datasets', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--sequence_length', type=int, default=50) # AE
    parser.add_argument('--validation_split', type=float, default=0.2) # AE
    parser.add_argument('--patience', type=int, default=6) # AE

    return parser.parse_args()

def read_train_data(training_dir, num_datasets):
    train_dfs = [pd.read_csv(os.path.join(training_dir, 'train-{}.csv'.format(i))) for i in range(num_datasets)]
    return train_dfs

def read_test_data(training_dir, num_datasets):
    test_dfs = [pd.read_csv(os.path.join(training_dir, 'test-{}.csv'.format(i))) for i in range(num_datasets)]
    return test_dfs

def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,142),(50,192)
    # 0 50 (start stop) -> from row 0 to row 50
    # 1 51 (start stop) -> from row 1 to row 51
    # 2 52 (start stop) -> from row 2 to row 52
    # ...
    # 141 191 (start stop) -> from row 141 to 191
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]
        
def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[seq_length:num_elements, :]

def rec_plot(s, eps=0.10, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z

def get_dataset(train_df, test_df, sequence_length):
    # NOTE: Skipping processing besides labels which are included in this page
    # see: https://github.com/awslabs/predictive-maintenance-using-machine-learning/blob/master/source/notebooks/sagemaker_predictive_maintenance/preprocess.py

    ### ADD NEW LABEL TRAIN ###
    w1 = 45
    w0 = 15
    train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0 )
    train_df['label2'] = train_df['label1']
    train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2

    ### ADD NEW LABEL TEST ###
    test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0 )
    test_df['label2'] = test_df['label1']
    test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2

    ### DROP NA DATA ###
    train_df = train_df.dropna(axis=1)
    test_df = test_df.dropna(axis=1)

    ### SEQUENCE COL: COLUMNS TO CONSIDER ###
    sequence_cols = []
    for col in train_df.columns:
        if col[0] == 's':
            sequence_cols.append(col)
    #sequence_cols.append('cycle_norm')
    logging.info('Sequence Cols: {}'.format(sequence_cols))
    
    ### GENERATE X TRAIN TEST ###
    x_train, x_test = [], []
    for engine_id in train_df.id.unique():
        for sequence in gen_sequence(train_df[train_df.id==engine_id], sequence_length, sequence_cols):
            x_train.append(sequence)
        for sequence in gen_sequence(test_df[test_df.id==engine_id], sequence_length, sequence_cols):
            x_test.append(sequence)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    logging.info("X_Train shape: {}".format(x_train.shape))
    logging.info("X_Test shape: {}".format(x_test.shape))
    
    ### GENERATE Y TRAIN TEST ###
    y_train, y_test = [], []
    for engine_id in train_df.id.unique():
        for label in gen_labels(train_df[train_df.id==engine_id], sequence_length, ['label2'] ):
            y_train.append(label)
        for label in gen_labels(test_df[test_df.id==engine_id], sequence_length, ['label2']):
            y_test.append(label)

    y_train = np.asarray(y_train).reshape(-1,1)
    y_test = np.asarray(y_test).reshape(-1,1)

    ### ENCODE LABEL ###
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    logging.info("y_train shape: {}".format(y_train.shape))
    logging.info("y_test shape: {}".format(y_test.shape))
        
    ### TRANSFORM X TRAIN TEST IN IMAGES ###
    x_train_img = np.apply_along_axis(rec_plot, 1, x_train).astype('float16')
    logging.info("x_train_image shape: {}".format(x_train_img.shape))

    x_test_img = np.apply_along_axis(rec_plot, 1, x_test).astype('float16')
    logging.info("x_test_image shape: {}".format(x_test_img.shape))

    return x_train_img, y_train, x_test_img, y_test

def fit_model(x_train_img, y_train, batch_size=512, epochs=25, validation_split=0.2, patience=6):
    input_shape = x_train_img.shape[1:]
    logging.info("Input shape: {}".format(input_shape))
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    logging.info(model.summary())
    
    ### FIT ###
    tf.random.set_seed(33)
    np.random.seed(33)
    random.seed(33)

    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1, 
        inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(), 
        config=session_conf
    )
    tf.compat.v1.keras.backend.set_session(sess)

    es = EarlyStopping(monitor='val_accuracy', mode='auto', restore_best_weights=True, verbose=1, patience=patience)

    model.fit(x_train_img, y_train, batch_size=batch_size, epochs=epochs, callbacks=[es],
              validation_split=validation_split, verbose=2)              
    
    ### EVAL ###
    logging.info('Evaluate: {}'.format(model.evaluate(x_test_img, y_test, verbose=2)))
    
    logging.info(classification_report(np.where(y_test != 0)[1], model.predict_classes(x_test_img)))
    return model

if __name__ == '__main__':
    logging = get_logger(__name__)
    logging.info('numpy version:{} Tensorflow version::{}'.format(np.__version__, tf.__version__))
    args = parse_args()

    # Read the first dataset
    train_df = read_train_data(args.training_dir, args.num_datasets)[0]
    test_df = read_test_data(args.training_dir, args.num_datasets)[0]
    
    # Get the training dataset as an image
    x_train_img, y_train, x_test_img, y_test = get_dataset(train_df, test_df, args.sequence_length)
    
    model = fit_model(x_train_img, y_train, 
              batch_size=args.batch_size, 
              epochs=args.epochs, 
              validation_split=args.validation_split,
              patience=args.patience)
    
    logging.info('saving model to: {}...'.format(args.model_dir))
    model.save(os.path.join(args.sm_model_dir, '000000001'))