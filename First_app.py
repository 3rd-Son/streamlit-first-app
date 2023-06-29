import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from PIL import Image

st.set_option("deprecation.showPyplotGlobalUse", False)

# Setting the title
st.title("SKlearn Dataset with Streamlit")

image = Image.open("ml image.jpeg")
st.image(image, use_column_width=True)

# setting the sutitle

st.write(
    """
         ##### **A simple data app with Streamlit to showcase my `ML Models`**
         """
)

st.write(""" #### Let's explore the different classifiers and datasets""")

dataset_name = st.sidebar.selectbox(
    "Select a Dataset", ("Breast Cancer", "Iris", "Wine")
)

classifier_name = st.sidebar.selectbox("Select a Classifier", ("SVM", "KNN"))


def get_dataset(name):
    data = None
    if name == "Iris":
        st.write("## ***This this is the sklearn `Iris` dataset***")
        data = datasets.load_iris()
    elif name == "Wine":
        st.write("## ***This this is the sklearn `Wine` dataset***")
        data = datasets.load_wine()
    else:
        st.write("## ***This this is the sklearn `breast cancer` dataset***")
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x, y


x, y = get_dataset(dataset_name)
st.dataframe(x)
st.write("Shape of your dataset:", x.shape)
st.write("Unique target variables:", len(np.unique(y)))

st.write("`BOXPLOT` OF THE DATASET")
fig = plt.figure()
sns.boxplot(data=x, orient="h")
st.pyplot()

st.write("`HISTOGRAM` OF THE DATASET")
plt.hist(x)
st.pyplot()


# sns.pairplot(pd.DataFrame(x), diag_kind="kde")
# st.pyplot()
random_state_no = st.sidebar.slider("Choose a random state", 1, 10)


# BUILDING THE ALGORITHMS
def add_parameter(name_of_clf):
    params = dict()
    if name_of_clf == "SVM":
        C = st.sidebar.slider("C", 0.01, 15.0)
        params["C"] = C
    else:
        name_of_clf == "KNN"
        k = st.sidebar.slider("k", 1, 15)
        params["k"] = k
    return params


params = add_parameter(classifier_name)


# ACCESSING THE CLASSIFIER
def get_classifier(name_of_clf, params):
    clf = None
    if name_of_clf == "SVM":
        clf = SVC(C=params["C"])
    elif name_of_clf == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["k"])
    else:
        st.warning("you didn't select any option, please select at least one algo")
    return clf


clf = get_classifier(classifier_name, params)


clf = get_classifier(classifier_name, params)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=random_state_no
)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

st.write(y_pred)

accuracy = accuracy_score(y_test, y_pred)

st.sidebar.write("classifier_name:", classifier_name)
st.sidebar.write("Acccuray for your model is:", accuracy)

st.success("Predicted successfuly")
st.markdown(
    "`Follow me on` [Twitter](https://twitter.com/SaxVictory) [LinkedIn](https://www.linkedin.com/in/victory-nnaji-8186231b7/) [Github](https://github.com/Vic3sax)"
)
