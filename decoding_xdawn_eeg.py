"""
.. _ex-xdawn-decoding:

============================
XDAWN Decoding From EEG data
============================

ERP decoding with Xdawn :footcite:`RivetEtAl2009,RivetEtAl2011`. For each event
type, a set of spatial Xdawn filters are trained and applied on the signal.
Channels are concatenated and rescaled to create features vectors that will be
fed into a logistic regression.
"""
# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from mne import io, pick_types, read_events, Epochs, EvokedArray
from mne.datasets import sample
from mne.preprocessing import Xdawn
from mne.decoding import Vectorizer
import scipy
from keras.models import Sequential
from keras.layers import Dense 
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import UpSampling1D

from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed



#print(__doc__)
#with numpy==1.19.5

data_path = sample.data_path()

###############################################################################
# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0.1, 0.3
event_id = {'Auditory/Left': 1, 'Auditory/Right': 2,
            'Visual/Left': 3, 'Visual/Right': 4}
n_filter = 3
# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname, preload=True)
#print(raw)
n_time_samps = raw.n_times
time_secs = raw.times
ch_names = raw.ch_names
n_chan = len(ch_names)  # note: there is no raw.n_channels attribute

print('the (cropped) sample data object has {} time samples and {} channels.'
      ''.format(n_time_samps, n_chan))
print('The last time sample is at {} seconds.'.format(time_secs[-1]))
print('The first few channel names are {}.'.format(', '.join(ch_names[:3])))
print()  # insert a blank line in the output

# some examples of raw.info:
print('bad channels:', raw.info['bads'])  # chs marked "bad" during acquisition
print(raw.info['sfreq'], 'Hz')            # sampling frequency
print(raw.info['description'], '\n')      # miscellaneous acquisition info

print(raw.info)
#raw.plot(duration=20)
#raw.plot_psd()
raw.filter(1, 20, fir_design='firwin')
#raw.plot(duration=20)
#raw.plot_psd()
events = read_events(event_fname)

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

epochs = Epochs(raw, events, event_id, tmin, tmax, proj=False,
                picks=picks, baseline=None, preload=True,
                verbose=False)

# Create classification pipeline
clf = make_pipeline(Xdawn(n_components=n_filter),
                    Vectorizer(),
                    MinMaxScaler(),
                    LogisticRegression(penalty='l1', solver='liblinear',
                                       multi_class='auto'))

# Get the labels
labels = epochs.events[:, -1]

# Cross validator
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Do cross-validation
preds = np.empty(len(labels))
for train, test in cv.split(epochs, labels):
    clf.fit(epochs[train], labels[train])
    preds[test] = clf.predict(epochs[test])

# Classification report
target_names = ['aud_l', 'aud_r', 'vis_l', 'vis_r']
report = classification_report(labels, preds, target_names=target_names)
print(report)

# Normalized confusion matrix
cm = confusion_matrix(labels, preds)
cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

# Plot confusion matrix
fig, ax = plt.subplots(1)
im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
ax.set(title='Normalized Confusion matrix')
fig.colorbar(im)
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)
fig.tight_layout()
ax.set(ylabel='True label', xlabel='Predicted label')

###############################################################################
# The ``patterns_`` attribute of a fitted Xdawn instance (here from the last
# cross-validation fold) can be used for visualization.

fig, axes = plt.subplots(nrows=len(event_id), ncols=n_filter,
                         figsize=(n_filter, len(event_id) * 2))
fitted_xdawn = clf.steps[0][1]
tmp_info = epochs.info.copy()
tmp_info['sfreq'] = 1.
for ii, cur_class in enumerate(sorted(event_id)):
    cur_patterns = fitted_xdawn.patterns_[cur_class]
    pattern_evoked = EvokedArray(cur_patterns[:n_filter].T, tmp_info, tmin=0)
    pattern_evoked.plot_topomap(
        times=np.arange(n_filter),
        time_format='Component %d' if ii == 0 else '', colorbar=False,
        show_names=False, axes=axes[ii], show=False)
    axes[ii, 0].set(ylabel=cur_class)
fig.tight_layout(h_pad=1.0, w_pad=1.0, pad=0.1)

sampling_freq = raw.info['sfreq']
fs=sampling_freq
start_stop_seconds = np.array([0, 20])
start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)
channel_index = 367
print(raw.ch_names[367])
name=raw.ch_names[367]
raw_selection = raw[channel_index, start_sample:stop_sample]
print(raw_selection)
xb = raw_selection[1]
yb = raw_selection[0].T
plt.figure('Плохой')
lines=plt.plot(xb, yb)
plt.legend(lines,name)
plt.grid()
print(n_chan)
name=raw.ch_names[366]
raw_selection = raw[channel_index, start_sample:stop_sample]

xg = raw_selection[1]
yg = raw_selection[0].T



yc=scipy.fft.ifft(scipy.fft.fft(yb)*scipy.fft.fft(yg))
print(yc.shape)
#yc=np.resize(yc,yc.shape[0])

plt.figure('Хороший')
lines=plt.plot(xg, yg)
plt.legend(lines,name)
plt.grid()

plt.figure('Взаимокорреляционная функция')
lines=plt.plot(yc)

plt.grid()


print(epochs)
ycn=yc/10
ycn=np.where(np.abs(ycn)!=1,ycn,0)
yc1=10*ycn
ydc=np.real(scipy.fft.ifft(scipy.fft.fft(yc1)/scipy.fft.fft(yg)))
'''
t=np.linspace(0,len(ydc)/fs,len(ydc))
x_train=np.column_stack((ydc,t,t))
#x_train=keras.utils.normalize(x_train) 
# this is the size of our encoded representations

encoding_dim = 3


input = Input(shape=(3,))
encoded = Dense(encoding_dim, activation='relu')(input)
decoded = Dense(3, activation='tanh')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(x_train, x_train,epochs=500,batch_size=4)

ydcp,t1,t2=np.hsplit(autoencoder.predict(x_train), 3)
#print(ydcp)

sequence = np.array(ydc[0:1000])
# reshape input into [samples, timesteps, features]
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
model.add(RepeatVector(n_in))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(sequence, sequence, epochs=300, verbose=0)
# connect the encoder LSTM as the output layer
model = Model(inputs=model.inputs, outputs=model.layers[0].output)
#plot_model(model, show_shapes=True, to_file='lstm_encoder.png')
# get the feature vector for the input sequence
yhat = model.predict(sequence)
print(yhat.shape)
yhat=yhat.reshape(yhat.shape[1])
print(yhat.shape)
#print(yhat)
t1=xg[0:1000]
plt.figure('Prediction')
plt.plot(t1,yhat)
plt.grid()
'''

seq_in = np.array(ydc[0:100])
in1=seq_in
print(in1.shape)
# reshape input into [samples, timesteps, features]
n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))
# prepare output sequence
seq_out = seq_in[:, 1:, :]
n_out = n_in - 1
# define encoder
visible = Input(shape=(n_in,1))
encoder = LSTM(100, activation='elu')(visible)
# define reconstruct decoder
decoder1 = RepeatVector(n_in)(encoder)
decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(1))(decoder1)
# define predict decoder
decoder2 = RepeatVector(n_out)(encoder)
decoder2 = LSTM(100, activation='selu', return_sequences=True)(decoder2)
decoder2 = TimeDistributed(Dense(1))(decoder2)
# tie it together
model = Model(inputs=visible, outputs=[decoder1, decoder2])
model.compile(optimizer='Adam', loss='mse')
model.summary()
#plot_model(model, show_shapes=True, to_file='composite_lstm_autoencoder.png')
# fit model
model.fit(seq_in, [seq_in,seq_out], epochs=300, verbose=0)
# demonstrate prediction
yhat = model.predict(seq_in, verbose=0)
out1=np.asarray(yhat[0])
out1=out1.reshape(in1.shape[0])
in1=in1.reshape(in1.shape[0])
t1=xg[0:100]
plt.figure('Prediction')
plt.plot(t1,out1)
plt.grid()

plt.figure('Разность предсказанного...')
plt.plot(t1,np.subtract(in1,out1))
plt.grid()

plt.figure('in')
plt.plot(t1,in1)
plt.grid()



print(epochs)
plt.figure('Сигнал после фильтрации')
plt.plot(xg[0:1000],ydc[0:1000])
plt.grid()

axis=0
ddof=0
a = np.asanyarray(ydc)
m = a.mean(axis)
sd = a.std(axis=axis, ddof=ddof)
x=np.std(a)
y=np.amax(a)
p=20*np.log10(y/x)
print('SNR after filtration')
print(p)
print()

axis=0
ddof=0
a = np.asanyarray(yb)
m = a.mean(axis)
sd = a.std(axis=axis, ddof=ddof)
x=np.std(a)
y=np.amax(a)
p=20*np.log10(y/x)
print(' SNR bad signal')
print(p)
print()

dydc=np.subtract(yb,ydc)
plt.figure('Разность...')
plt.plot(xg,dydc)
plt.grid()
#ydc=scipy.signal.deconvolve(signal=yc,divisor=window)

plt.show()

###############################################################################
# References
# ----------
# .. footbibliography::
