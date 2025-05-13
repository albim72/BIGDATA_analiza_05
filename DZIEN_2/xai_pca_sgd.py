#globalna interpretacja cech

import pandas as pd
import matplotlib.pyplot as plt

# Załaduj dane i dopasuj PCA jak wcześniej
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_breast_cancer()
X = data.data
feature_names = data.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
pca.fit(X_scaled)

# PCA komponenty
components_df = pd.DataFrame(pca.components_, columns=feature_names, index=['PC1', 'PC2', 'PC3'])
components_df.T.plot.bar(figsize=(14, 6), title="Wpływ cech oryginalnych na PCA")
plt.ylabel("Waga (ładunek wektora)")
plt.grid(True)
plt.tight_layout()
plt.show()


#shap

import shap

# Przekształcamy oryginalne dane treningowe do PCA
X_pca = pca.transform(X_scaled)

# Dopasuj model
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='hinge')
clf.fit(X_pca, data.target)

# Obiekt SHAP dla PCA
explainer = shap.Explainer(clf.predict, X_pca)
shap_values = explainer(X_pca[:100])  # pierwsze 100 próbek

# Wizualizacja: wartości SHAP
shap.plots.beeswarm(shap_values)

#lime

import lime
import lime.lime_tabular

from sklearn.model_selection import train_test_split
import numpy as np
# 1. Wczytanie danych
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
class_names = data.target_names

# 2. Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standaryzacja
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = np.vstack((X_train_scaled, X_test_scaled))

# 4. Redukcja wymiarów (PCA)
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 5. Klasyfikator – liniowy SVM (SGD)
clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
clf.fit(X_train_pca, y_train)

# 6. LIME – wyjaśnianie
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_scaled,
    feature_names=feature_names,
    class_names=class_names,
    verbose=True,
    mode='classification'
)

# 7. Wyjaśnienie konkretnej próbki
i = 0  # indeks próbki
exp = explainer.explain_instance(
    X_scaled[i],
    lambda x: clf.predict_proba(pca.transform(x))
)

# 8. Prezentacja wyników
exp.show_in_notebook(show_table=True)
fig = exp.as_pyplot_figure()
plt.show()
