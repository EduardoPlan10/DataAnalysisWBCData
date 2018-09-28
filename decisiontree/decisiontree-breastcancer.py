# -*- coding: utf-8 -*-

import itertools
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

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


b_cancer = load_breast_cancer()
X = b_cancer.data
y = b_cancer.target
class_names = b_cancer.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

classifier = tree.DecisionTreeClassifier()

y_pred = classifier.fit(X_train, y_train).predict(X_test)


cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=('maligno', 'benigno'), normalize=True,
                      title=(u'Matriz de confusão para árvore de decisão'))

plt.savefig('decisiontreebreastcancer.eps', format='eps', dpi=400,
            bbox_inches='tight')


dot_data = tree.export_graphviz(classifier, out_file=None,
                feature_names=b_cancer.feature_names,
                class_names=b_cancer.target_names,
                filled=True, rounded=True,
                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("breast cancer")
