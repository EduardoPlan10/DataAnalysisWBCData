from sklearn.datasets import load_breast_cancer
from sklearn import tree
import graphviz

b_cancer = load_breast_cancer()
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(breast_cancer.data, breast_cancer.target)
class_names = b_cancer.target_names


dot_data = tree.export_graphviz(classifier, out_file=None,
                feature_names=breast_cancer.feature_names,
                class_names=breast_cancer.target_names,
                filled=True, rounded=True,
                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("breast cancer")
