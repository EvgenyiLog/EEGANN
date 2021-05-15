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



print(__doc__)

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
print(raw)
n_time_samps = raw.n_times
time_secs = raw.times
ch_names = raw.ch_names
n_chan = len(ch_names)  # note: there is no raw.n_channels attribute
print(n_chan)
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
plt.figure()
lines=plt.plot(xb, yb)
plt.legend(lines,name)
plt.grid()
print(n_chan)
name=raw.ch_names[366]
raw_selection = raw[channel_index, start_sample:stop_sample]

xg = raw_selection[1]
yg = raw_selection[0].T



yc=scipy.signal.convolve(yb,yg)
print(yc.shape)
yc=np.resize(yc,yc.shape[0])

plt.figure()
lines=plt.plot(xg, yg)
plt.legend(lines,name)
plt.grid()

plt.figure()
lines=plt.plot(yc)

plt.grid()

t=np.linspace(0,len(yc)/fs,len(yc))
x_train=np.column_stack((yc,t,t))
#x_train=keras.utils.normalize(x_train) 
# this is the size of our encoded representations

encoding_dim = 3


input = Input(shape=(3,))
encoded = Dense(encoding_dim, activation='relu')(input)
decoded = Dense(3, activation='linear')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(x_train, x_train,epochs=1000,batch_size=4)

yc1,t1,t2=np.hsplit(autoencoder.predict(x_train), 3)

plt.figure()
plt.plot(yc)
plt.grid()




plt.show()

###############################################################################
# References
# ----------
# .. footbibliography::
