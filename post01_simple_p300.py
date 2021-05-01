from pathlib import Path
from importlib import reload

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import moabb.datasets


matplotlib.style.use('seaborn')
plt.rcParams.update({'font.size': 15})

sampling_rate = 512

m_dataset = moabb.datasets.bi2013a(
    NonAdaptive=True,
    Adaptive=True,
    Training=True,
    Online=True,
)

m_dataset.download()  
m_data = m_dataset.get_data()


print(m_data[1]['session_1']['run_1'])
channels = m_data[1]['session_1']['run_1'].ch_names[:-1]


raw_dataset = []

for _, sessions in sorted(m_data.items()):
    eegs, markers = [], []
    for item, run in sorted(sessions['session_1'].items()):
        data = run.get_data()
        eegs.append(data[:-1])
        markers.append(data[-1])
    raw_dataset.append((eegs, markers))

raw_sample = raw_dataset[0][0][0]

print(len(raw_dataset), len(raw_dataset[0]), len(raw_dataset[0][0]), raw_dataset[0][0][0].shape)

del m_data
del m_dataset


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import transformers


reload(transformers)

decimation_factor = 10
final_rate = sampling_rate // decimation_factor
epoch_duration = 0.9 # seconds
labels_mapping = {33285.: 1, 33286.: 0}
captions = {0: 'empty', 1: 'target'}

eeg_pipe = make_pipeline(
    transformers.Decimator(decimation_factor),
    transformers.ButterFilter(sampling_rate // decimation_factor, 4, 0.5, 20),
    transformers.ChannellwiseScaler(StandardScaler()),
)
markers_pipe = transformers.MarkersTransformer(labels_mapping, decimation_factor)


for eegs, _ in raw_dataset:
    eeg_pipe.fit(eegs)


dataset = []
epoch_count = int(epoch_duration * final_rate)

for eegs, markers in raw_dataset:
    epochs = []
    labels = []
    filtered = eeg_pipe.transform(eegs)
    markups = markers_pipe.transform(markers)
    for signal, markup in zip(filtered, markups):
        epochs.extend([signal[:, start:(start + epoch_count)] for start in markup[:, 0]])
        labels.extend(markup[:, 1])
    dataset.append((np.array(epochs), np.array(labels)))


print(dataset[0][0].shape, dataset[0][0].dtype, dataset[0][1].shape)


del raw_dataset


epoch = eeg_pipe.fit_transform([raw_sample])[0]
plt.figure(figsize=(20, 15))
plt.plot(epoch.T + np.arange(len(epoch))*5)
plt.yticks([])
plt.xticks(np.arange(0, 16000, 1000), np.arange(0, 32, 1))
plt.xlabel('seconds', fontsize=15)
plt.ylabel('channels', fontsize=15)
plt.title('Filtered EEG signal of one run of the game')

epoch = dataset[0][0][0]
plt.figure(figsize=(20, 10))
plt.plot(epoch.T + np.arange(len(epoch)))
plt.yticks([])
plt.xticks(np.arange(0, 50, 10), np.arange(0, 1000, 200))
plt.xlabel('milliseconds', fontsize=15)
plt.ylabel('channels', fontsize=15)


all_epochs = np.concatenate([epochs for epochs, _ in dataset])


print(all_epochs.shape, all_epochs.dtype)


ll_labels = np.concatenate([labels for _, labels in dataset])

uniques, counts = np.unique(all_labels, return_counts=True)

plt.bar(uniques, counts, color=['r', 'g'])
plt.xticks(uniques)

print(counts[0] / counts[1])


def plot_by_labels(epochs, labels):
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    for label in (0, 1):
        mean = epochs[labels == label].mean(axis=(0, 1))
        std = epochs[labels == label].std(axis=(0, 1))

        axes[0].plot(mean, label=captions[label])

        axes[1].plot(mean, label=captions[label])
        axes[1].fill_between(np.arange(mean.size), mean-std, mean+std, alpha=0.25)

    axes[0].legend(fontsize=20)
    axes[0].set_ylim(-0.16, 0.26)
    axes[1].set_ylim(-0.8, 0.85)
    for i, title in enumerate(('Means', 'Means with Stds')):
        axes[i].set_title(title, fontsize=20)
        axes[i].set_xticklabels(np.arange(-200, 1000, 200))
        axes[i].set_xlabel('milliseconds')



plot_by_labels(all_epochs, all_labels)

plot_by_labels(dataset[6][0], dataset[6][1])


import visualizers  
reload(visualizers)

visualizers.plot_evoked_electrodes(dataset, final_rate, channels)

reload(visualizers)

visualizers.plot_evoked_map(dataset, final_rate, channels)


from sklearn.model_selection import GridSearchCV, PredefinedSplit

from mne.decoding import Vectorizer, CSP

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

from pyriemann.estimation import ERPCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from pyriemann.spatialfilters import Xdawn


scores = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')


# from https://eeg-notebooks.readthedocs.io/en/latest/visual_p300.html
# and https://mne.tools/dev/auto_examples/decoding/plot_decoding_csp_eeg.html

clfs = {
    'LR': (
        make_pipeline(Vectorizer(), LogisticRegression()),
        {'logisticregression__C': np.exp(np.linspace(-4, 4, 9))},
    ),
    'LDA': (
        make_pipeline(Vectorizer(), LDA(shrinkage='auto', solver='eigen')),
        {},
    ),
    'SVM': (
        make_pipeline(Vectorizer(), SVC()),
        {'svc__C': np.exp(np.linspace(-4, 4, 9))},
    ),
    'CSP LDA': (
        make_pipeline(CSP(), LDA(shrinkage='auto', solver='eigen')),
        {'csp__n_components': (6, 9, 13), 'csp__cov_est': ('concat', 'epoch')},
    ),
    'Xdawn LDA': (
        make_pipeline(Xdawn(2, classes=[1]), Vectorizer(), LDA(shrinkage='auto', solver='eigen')),
        {},
    ),
    'ERPCov TS LR': (
        make_pipeline(ERPCovariances(estimator='oas'), TangentSpace(), LogisticRegression()),
        {'erpcovariances__estimator': ('lwf', 'oas')},
    ),
    'ERPCov MDM': (
        make_pipeline(ERPCovariances(), MDM()),
        {'erpcovariances__estimator': ('lwf', 'oas')},
    ),
}


def crossvalidate_record(record, clfs=clfs, scores=scores):
    df = pd.DataFrame()
    for name, (clf, params) in clfs.items():
        cv = GridSearchCV(
            clf,
            params,
            scoring=scores,
            n_jobs=-1,
            iid=False,
            refit=False,
            cv=4,
        )
        cv.fit(record[0], record[1])
        headers = [
            name for name in cv.cv_results_.keys()
                if name.startswith('param_') or name.startswith('mean_test_') or name.startswith('std_test_')
        ]
        results = pd.DataFrame(cv.cv_results_)[headers]
        results['cassifier'] = name
        df = pd.concat((df, results), sort=False)
    return df.reindex(sorted(df.columns), axis=1)

crossvalidate_record(dataset[0]).sort_values('mean_test_f1', ascending=False)


def crossvalidate_dataset(dataset, clfs=clfs, scores=scores):
    res = {name: [] for name in scores}
    for record in dataset:
        df = crossvalidate_record(record, clfs, scores)
        for name in scores:
            res[name].append(df[f'mean_test_{name}'])

    final = df.copy()
    for name, values in res.items():
        values = np.array(values)
        final[f'mean_test_{name}'] = values.mean(axis=0)
        final[f'std_test_{name}'] = values.std(axis=0)
    return final.sort_values('mean_test_f1', ascending=False)


crossvalidate_dataset(dataset)

def transfer_validate(train_records: list, val_records: list, clfs=clfs, scores=scores):
    df = pd.DataFrame()
    
    train_epochs = np.concatenate([epochs for epochs, _ in train_records])
    train_labels = np.concatenate([labels for _, labels in train_records])
    val_epochs = np.concatenate([epochs for epochs, _ in val_records])
    val_labels = np.concatenate([labels for _, labels in val_records])
    splitter = PredefinedSplit([-1] * len(train_labels) + [0] * len(val_labels))
    epochs = np.concatenate((train_epochs, val_epochs))
    labels = np.concatenate((train_labels, val_labels))

    for name, (clf, params) in clfs.items():
        cv = GridSearchCV(
            clf,
            params,
            scoring=scores,
            n_jobs=-1,
            iid=False,
            refit=False,
            cv=splitter,
        )
        cv.fit(epochs, labels)
        headers = [
            name for name in cv.cv_results_.keys()
                if name.startswith('param_') or name.startswith('mean_test_') or name.startswith('std_test_')
        ]
        results = pd.DataFrame(cv.cv_results_)[headers]
        results['cassifier'] = name
        df = pd.concat((df, results), sort=False)
    return df.reindex(sorted(df.columns), axis=1)


transfer_validate(dataset[:1], dataset[10:15])
transfer_validate(dataset[:10], dataset[10:15])

plt.show()