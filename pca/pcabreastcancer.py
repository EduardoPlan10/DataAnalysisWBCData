# -*- coding: utf-8 -*-

from __future__ import print_function

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Prevista')


FIG_SIZE = (10, 7)

features, target = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                   random_state=0)

unscaled_clf = make_pipeline(PCA(n_components=2), GaussianNB())
unscaled_clf.fit(X_train, y_train)
y_pred_nonstandart = unscaled_clf.predict(X_test)

std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
std_clf.fit(X_train, y_train)
y_pred = std_clf.predict(X_test)

print('\nPrediction accuracy for the normal test dataset with PCA')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, y_pred_nonstandart)))

print('\nPrediction accuracy for the standardized test dataset with PCA')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, y_pred)))

# Extract PCA from pipeline
pca = unscaled_clf.named_steps['pca']
pca_std = std_clf.named_steps['pca']

# Show first principal componenets
print('\nPC 1 without scaling:\n', pca.components_[0])
print('\nPC 1 with scaling:\n', pca_std.components_[0])

# Scale and use PCA on X_train data for visualization.
scaler = std_clf.named_steps['standardscaler']
X_train_std = pca_std.transform(scaler.transform(X_train))

# visualize standardized vs. untouched dataset with PCA performed
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=FIG_SIZE)


for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax1.scatter(X_train[y_train == l, 0], X_train[y_train == l, 1],
                color=c,
                label='classe %s' % l,
                alpha=0.5,
                marker=m
                )

for l, c, m in zip(range(0, 3), ('blue', 'red', 'green'), ('^', 's', 'o')):
    ax2.scatter(X_train_std[y_train == l, 0], X_train_std[y_train == l, 1],
                color=c,
                label='classe %s' % l,
                alpha=0.5,
                marker=m
                )

ax1.set_title('Amostra de treinamento usando PCA')
ax2.set_title('Amostra de treinamento normalizada usando PCA')

for ax in (ax1, ax2):
    ax.set_xlabel('Primeiro componente principal')
    ax.set_ylabel('Segundo componente principal')
    ax.legend(loc='upper right')
    ax.grid()

plt.tight_layout()
plt.savefig('pca-componentanalysis-breastcancer.eps', format='eps', dpi=400, bbox_inches='tight')


cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
plt.figure()

title = 'Normalized confusion matrix'
plot_confusion_matrix(cnf_matrix, classes=('maligno', 'benigno'), normalize=True,
                      title=(u'Matriz de confusão para ACP (análise de componente principal'))

plt.savefig('PCAbreastcancernormalized.eps', format='eps', dpi=400, bbox_inches='tight')


