# -*- coding: utf-8 -*-

# -- Utility functions --

# convert hex string to unsigned int (es. FF00 -> 65280)
def hex2int_unsigned(hex_val):
    return int(hex_val,16)

# convert hex string to signed int (es. FF00 -> -256)
def hex2int_signed(hexval):
    bits = 16
    val = int(hexval, bits)
    if val & (1 << (bits-1)):
        val -= 1 << bits
    return val


# read text file into pandas DataFrame
def load_trc_file(file_name, skip_rows, n_rows = None):
    data_frame = pd.read_csv(
        file_name,
        delim_whitespace=True, 
        skiprows = skip_rows, 
        header=None, 
        names=["N","O","T","B","I","d","R","L","D0","D1","D2","D3","D4","D5","D6","D7"],
        nrows= n_rows,
        index_col = 'O',
        converters={
            'O': partial(pd.to_datetime,unit='ms',infer_datetime_format=True)
            #    'D0': partial(int, base=16),
        }
        
    )
    return data_frame

# extract posital values
def get_posital_data(data_frame, message_id):
    
    df1 = data_frame[data_frame["I"] == message_id]

    # convert angle from hex bytes to unsigned int
    df1['P2'] = (df1['D1'] + df1['D0']).apply(hex2int_unsigned)

    # resample at 10 milliseconds
    df1 = df1.resample("10L").ffill()

    return df1


def get_joystick_data(data_frame, message_id):
    # command = df[df["I"] == '02B2']
    df1 = data_frame[data_frame["I"] == message_id]

    # convert set points from hex strings to signed int
    df1['C2'] = (df1['D1'] + df1['D0']).apply(hex2int_signed)
    df1['C1'] = (df1['D3'] + df1['D2']).apply(hex2int_signed)

    # resample at 10 milliseconds
    df1 = df1.resample("10L").ffill()

    return df1


def merge_columns(cmd, pos, cmd_rolling_length = 1, pos_rolling_length = 1):

    min_rows = min(cmd.shape[0], pos.shape[0])

    cmd = cmd[['C1','C2']].iloc[0:min_rows].copy()
    cmd = cmd.rolling(cmd_rolling_length).mean()

    pos = pos[['P2']].iloc[0:min_rows].copy()
    pos = pos.rolling(pos_rolling_length).mean()

    df1 = pd.merge(cmd, pos, left_index=True, right_index=True)
    df1 = df1.dropna()

    return df1

def plot_joystick_and_position(df1):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    df1.plot(y=['P2'], ax=ax)
    df1.plot(y=["C1", "C2"], ax=ax2, ls="--")

    plt.show()


# constants
C1_BASE = 3  # start from columns 3
C1_STEP = 3  # take 3 steps back
C1_SIZE = 33 # was 33 in the ETH model

C2_BASE = C1_BASE+C1_SIZE  # start from columns 3+5
C2_STEP = 3  # take 3 steps back
C2_SIZE = 33 # was 33 in the ETH model


V_BASE = C1_BASE+C1_SIZE+C2_SIZE 
V_STEP = 1 # take 1 step back
V_SIZE = 10 # was 10 in the ETH model

def augment_columns(df3):
    # create columns with past commands
    for i in range(1,C1_SIZE+1):
        df3[C1_BASE+i] = df3['C1'].shift(i*C1_STEP)
        df3.rename(columns={C1_BASE+i:'C1-'+ str(i*C1_STEP)}, inplace=True)

    for i in range(1,C2_SIZE+1):
        df3[C2_BASE+i] = df3['C2'].shift(i*C2_STEP)
        df3.rename(columns={C2_BASE+i:'C2-'+ str(i*C2_STEP)}, inplace=True)

    # create velocities from positions
    df3['V2'] = df3['P2']-df3['P2'].shift(1)

    # filter velocities
    df3['V2'] = df3['V2'].rolling(velocity_rolling_length).mean()

    # create delta velocities from velocities
    # df3['DV2'] = df3['V2']-df3['V2'].shift(1)

    # create columns with past velocities
    for i in range(1,V_SIZE+1):
        df3[V_BASE+i] = df3['V2'].shift(i*V_STEP)
        df3.rename(columns={V_BASE+i:'V2-'+ str(i*V_STEP)}, inplace=True) 
    
    df3 = df3.fillna(0)

    return df3


# utility function
def plot_loss(hist, ymax=100):
  plt.plot(hist.history['loss'], label='loss')
  plt.plot(hist.history['val_loss'], label='val_loss')
  plt.ylim([0, ymax])
  plt.xlabel('Epoch')
  plt.ylabel('Error [y]')
  plt.legend()
  plt.grid(True)


def plot_predictions_vs_labels(test_preds, test_lbls, max_value=30):
    a = plt.axes(aspect='equal')
    plt.scatter(test_lbls, test_preds)
    plt.xlabel('True Values [y]')
    plt.ylabel('Predictions [y]')
    lims = [0, max_value]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)


def plot_prediction_error(err):
    plt.hist(err, bins=25)
    plt.xlabel('Prediction Error [y]')
    _ = plt.ylabel('Count')



# -- Load training File --

# ## Load training file


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


# adjust these parameters

# data collection
# fname = "../input/posital-2-braccio/Posital 2 braccio.trc"
#fname = "../input/gru-dpx-posital-1-braccio-training/Gru_DPX_Posital_1_braccio_Training.trc"
train_fname = "/data/workspace_files/Gru_DPX_Posital_1_braccio_Test_Andrea.trc"

train_skip_rows = 20
train_n_rows = None

posital_message_id='0182'
joystick_message_id='02B1'

# data filter
joystick_rolling_length = 10
position_rolling_length = 10
velocity_rolling_length = 1

# keras model
labels_column_names = ['V2']
output_dimension = len(labels_column_names)
model_filename= './model_1'

# read text file into pandas DataFrame
df = load_trc_file(file_name=train_fname, skip_rows=train_skip_rows, n_rows=train_n_rows)
print(df)

# extract posital rows
posital = get_posital_data(df, message_id=posital_message_id)

# extract joystick rows
joystick = get_joystick_data(df, message_id=joystick_message_id)

train_df2 = merge_columns(joystick, posital, joystick_rolling_length, position_rolling_length)
train_df2.plot()
plot_joystick_and_position(train_df2)

train_df2.describe()

# -- Train KERAS Model --

# ## Train KERAS Model


# create additional columns
train_df3 = train_df2.copy()
keras_dataset = augment_columns(train_df3)

# split dataset in train and test
train_dataset = keras_dataset.sample(frac=0.8, random_state=0)
test_dataset = keras_dataset.drop(train_dataset.index)

# extract features and labels into separate arrays
train_features = train_dataset.copy()
train_features = train_features.drop(labels_column_names, axis=1)
train_labels = train_dataset.loc[:, labels_column_names]

test_features = test_dataset.copy()
test_features = test_features.drop(labels_column_names, axis=1)
test_labels = test_dataset.loc[:, labels_column_names]

print('\n', test_labels.columns, '\n', test_features.columns)

# show train_dataset mean and std
train_dataset.describe().transpose()

# create a Normalization layer
normalizer = tf.keras.layers.Normalization(axis=-1)  # many variables
# normalizer = tf.keras.layers.Normalization(input_shape=[1,], axis=None) #one variable

# set mean and std for Normalization layer
normalizer.adapt(np.array(train_features))
# print(normalizer.mean.numpy())

# create keras model

dnn_model = keras.Sequential([
      normalizer,
      layers.Dense(units=128, activation='relu'),
      layers.Dense(units = 128, activation='relu'),
      layers.Dense(units=128, activation='relu'),
      layers.Dense(output_dimension)
  ])

dnn_model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
dnn_model.summary()

# train keras model

history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, 
    epochs=10
)

plot_loss(hist= history,ymax=2)

# evaluate model loss with test_data 
test_results = {}
test_results = dnn_model.evaluate(test_features, test_labels, verbose=0)
test_results

# compare test predictions with test labels
test_predictions = dnn_model.predict(test_features)
plot_predictions_vs_labels(test_predictions, test_labels, max_value=20)

# evaluate predictions error
error = test_predictions - test_labels
plot_prediction_error(error)

plt.plot(test_predictions), train_df3[labels_column_names].plot()

plt.plot(test_labels - test_predictions)

dnn_model.save(model_filename)
reloaded = tf.keras.models.load_model(model_filename)

# -- Load test file --

# ## Load test file


# adjust these parameters

# data collection
test_fname = "/data/workspace_files/Gru_DPX_Posital_1_braccio_Test_Moussa.trc"
test_skip_rows = 20
test_n_rows = None

# read text file into pandas DataFrame
df = load_trc_file(file_name=test_fname, skip_rows=test_skip_rows, n_rows=test_n_rows)
print(df)

# extract posital rows
test_pos = get_posital_data(df, message_id=posital_message_id)

# extract joystick rows
test_joy = get_joystick_data(df, message_id=joystick_message_id)

test_df2 = merge_columns(test_joy, test_pos, joystick_rolling_length, position_rolling_length)
test_df2.plot()
plot_joystick_and_position(test_df2)

# -- Evaluate test file --

# ## Evaluate test file


test_df2

# create additional columns
test_df3 = test_df2.copy()
keras_dataset = augment_columns(test_df3)
print(keras_dataset)

# use the whole file for testing, there is no training phase

# split dataset in train and test
test_dataset = keras_dataset.copy()

# extract features and labels into separate arrays
test_features = test_dataset.copy()
test_features = test_features.drop(labels_column_names, axis=1)
test_labels = test_dataset.loc[:, labels_column_names]

# evaluate model loss with test_data 
test_results = {}
test_results = dnn_model.evaluate(test_features, test_labels, verbose=0)
test_results

test_features

# compare test predictions with test labels
test_predictions = dnn_model.predict(test_features)
plot_predictions_vs_labels(test_predictions, test_labels, max_value=20)

# evaluate predictions error
error = test_predictions - test_labels
plot_prediction_error(error)

plt.plot(test_predictions),test_df3[labels_column_names].plot()

plt.plot(test_labels - test_predictions)

# -- Predict --

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


pred_model_filename = 'model_1'
velocity_rolling_length = 1

#cmd_1 = [0, 0, 0, 0, 0, 0, 1000, 1000, 1000, 1000, -1000, -1000, -1000, -1000]
cmd_1 = 1000 * np.ones(1000)
for i in range(0,500): cmd_1[i] = 0
for i in range(600,1000): cmd_1[i] = 0

cmd_2 = np.zeros_like(cmd_1)
pos_2 = np.zeros_like(cmd_1)
  

# Calling DataFrame constructor after zipping both lists, with columns specified
pred_df2 = pd.DataFrame(list(zip(cmd_1, cmd_2, pos_2)),
               columns =['C1', 'C2', 'P2'])
#pred_df2

reloaded_model = tf.keras.models.load_model(pred_model_filename)
#reloaded_model.summary()

pred_df3 = pred_df2.copy()
pred_keras_dataset = augment_columns(pred_df3)
pred_keras_dataset

pred_dataset = pred_keras_dataset.copy()

# extract features and labels into separate arrays
pred_features = pred_dataset.copy()
pred_features = pred_features.drop(labels_column_names, axis=1)
pred_labels = pred_dataset.loc[:, labels_column_names]


rr_base = 0
rr_size = 10
rr = range(rr_base, rr_base + rr_size)

pV2 = np.zeros(rr_base + rr_size)


for pred_step in rr:

    # create input dataframe
    pf = pred_features.iloc[range(pred_step,pred_step+1)]
    # print(pf)

    pf = np.zeros_like(pf) 
    pf[0][2] = 10000
    pred_features.P2.iloc[pred_step] = 10000


    # predict velocity
    pV2[pred_step] = reloaded_model.predict(pf)


    next_C1 = cmd_1[pred_step + 1]
    next_C2 = cmd_2[pred_step + 1]
    next_P2 = float(pred_features.P2.iloc[pred_step] + pV2[pred_step])

    pred_features['C1'].iloc[pred_step+1] = next_C1
    pred_features['C2'].iloc[pred_step+1] = next_C2
    pred_features['P2'].iloc[pred_step+1] = next_P2

    pred_features['V2-1'].iloc[pred_step+1] = pV2[pred_step]



pred_features.iloc[rr]

#plt.plot(pred_predictions)
 
pV2



pred_features['P2'].iloc[rr].plot()

