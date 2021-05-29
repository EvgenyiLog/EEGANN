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

import pywt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler



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


#matched filter
yc=scipy.fft.ifft(scipy.fft.fft(yb)*scipy.fft.fft(yg))


#plot signal
plt.figure('Хороший')
lines=plt.plot(xg, yg)
plt.legend(lines,name)
plt.grid()

#plot h matched
plt.figure('Взаимокорреляционная функция')
lines=plt.plot(yc)

plt.grid()


print(epochs)
#mathed filtration
ycn=yc/10
ycn=np.where(np.abs(ycn)!=1,ycn,0)
yc1=10*ycn
ydc=np.real(scipy.fft.fftshift(scipy.fft.ifft(scipy.fft.fft(yc1)/scipy.fft.fft(yg))))

#input wavelet decompasition
in1=ydc

#wavelet decompasition using dwt
CA11,CD11,CD10,CD9,CD8,CD7,CD6,CD5,CD4,CD3,CD2,CD1=pywt.wavedec(ydc,wavelet=pywt.Wavelet('db2'),level=11)

#predict 10 level dwt decompasition coefficient detalisation
CD10P=[]
for i in range(1):
    t=np.linspace(0,len(CD10[i,:])/fs,len(ydc))
    ca11=np.resize(CD10,CD10.shape[0])
    x_train=np.column_stack((ca11,t,t))
    x_train=np.asarray(x_train)
   

    encoding_dim = 3


    input = Input(shape=(3,))
    encoded = Dense(encoding_dim, activation='relu')(input)
    decoded = Dense(3, activation='relu')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mse')

    autoencoder.fit(x_train, x_train,epochs=500,batch_size=4)

    y,t1,t2=np.hsplit(autoencoder.predict(x_train), 3)
    CD10P=np.append(CD10P,y)
    
    CD10P=np.vstack((CD10P,CD10P))
    
    CD10P=np.resize(CD10P,(CD10.shape[0],CD10.shape[1]))

#recover signal
out11=pywt.idwt(CA11,CD11,wavelet=pywt.Wavelet('db2'))
out10=pywt.idwt(CA11,CD10P,wavelet=pywt.Wavelet('db2'))
out9=pywt.idwt(CA11,CD9,wavelet=pywt.Wavelet('db2'))
out8=pywt.idwt(CA11,CD8,wavelet=pywt.Wavelet('db2'))
out7=pywt.idwt(CA11,CD7,wavelet=pywt.Wavelet('db2'))
out6=pywt.idwt(CA11,CD6,wavelet=pywt.Wavelet('db2'))
out5=pywt.idwt(CA11,CD5,wavelet=pywt.Wavelet('db2'))
out4=pywt.idwt(CA11,CD4,wavelet=pywt.Wavelet('db2'))
out3=pywt.idwt(CA11,CD3,wavelet=pywt.Wavelet('db2'))
out20=pywt.idwt(CA11,CD2,wavelet=pywt.Wavelet('db2'))
out1=pywt.idwt(CA11,CD1,wavelet=pywt.Wavelet('db2'))

#filtration 10 level dwt decompasition using polinomial regression with Ridge regularization
out2end=np.mean(np.add(out11,out10),axis=1)
t2=np.linspace(0,len(out2end)/fs,len(out2end))
X = t2[:, np.newaxis]
X_plot = t2[:, np.newaxis]
model = make_pipeline(PolynomialFeatures(20), Ridge())
out2end=out2end-np.mean(out2end)
model.fit(X,out2end)
out2end1=model.predict(X_plot)
#recover signal
out2=np.add(out2end1,np.mean(out9,axis=1))
out2=np.add(out2,np.mean(out8,axis=1))
out2=np.add(out2,np.mean(out7,axis=1))
out2=np.add(out2,np.mean(out6,axis=1))
out2=np.add(out2,np.mean(out5,axis=1))
out2=np.add(out2,np.mean(out4,axis=1))
out2=np.add(out2,np.mean(out3,axis=1))
out2=np.add(out2,np.mean(out20,axis=1))
out2=np.add(out2,np.mean(out1,axis=1))
out2end=out2
out2end=out2end-np.mean(out2end)
t2=np.linspace(0,len(out2)/fs,len(out2))

#print(out2end.shape)
plt.figure('out2')
plt.plot(t2,out2end,'r-')
plt.grid()



in1=ydc
#print(in1.shape)
in1=np.resize(in1,in1.shape[0])
#print(out2end.shape)
t1=np.linspace(0,len(in1)/fs,len(in1))

plt.figure('Разность после согласованного фильтра и конечного отфильтрованного с помощью ДВП,автоэнокодера,')
plt.plot(t1,np.subtract(in1,out2end))
plt.grid()
print('Mean error')
print(np.mean(np.subtract(in1,out2end)))
print()
#signal after mathed filter
in1=ydc
plt.figure('in')
plt.plot(t1,in1)
plt.grid()


axis=0
ddof=0
a = np.asanyarray(ydc)
m = a.mean(axis)
sd = a.std(axis=axis, ddof=ddof)
x=np.std(a)
y=np.amax(a)
p=20*np.log10(y/x)
print('SNR after  matched filtration')
print(p)
print()

s=np.power(ydc,2)
s2=np.sum(s)
i=-s2*np.log2(s2)
print('informativaty after  matched filtration')
print(i)
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

s=np.power(yb,2)
s2=np.sum(s)
i=-s2*np.log2(s2)
print('informativaty bad signal')
print(i)
print()

axis=0
ddof=0
a = np.asanyarray(out2end)
m = a.mean(axis)
sd = a.std(axis=axis, ddof=ddof)
x=np.std(a)
y=np.amax(a)
p=20*np.log10(y/x)
print(' SNR after filtration and prediction')
print(p)
print()

s=np.power(out2end,2)
s2=np.sum(s)
i=-s2*np.log2(s2)
print('informativaty out2end ')
print(i)
print()





dydc=np.subtract(yb,ydc)
plt.figure('Разность плохого и фильтрованнного согласованным фильтром')
plt.plot(xg,dydc)
plt.grid()
#ydc=scipy.signal.deconvolve(signal=yc,divisor=window)

plt.show()

###############################################################################
# References
# ----------
# .. footbibliography::
