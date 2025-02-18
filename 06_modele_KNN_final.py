import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.combine import SMOTEENN
from sklearn.neighbors import KNeighborsClassifier
import time
import pickle

# 📌 Définition du modèle et des hyperparamètres
model = KNeighborsClassifier(metric='euclidean', n_neighbors=3, weights='distance')

# 📌 Charger les données
print("🔄 Chargement des données...")
start_time = time.time()

df = pd.read_csv("06_ETL_entrainement.csv")

# Transformation des noms des colonnes
df.columns = ['id_trans', 'etape', 'type', 'montant', 'id_orig', 'old_solde_exp',
              'new_solde_exp', 'id_dest', 'old_solde_dest', 'new_solde_dest', 'fraude']

# Création de variables
df['diff_solde_orig'] = abs(df['old_solde_exp'] - df['new_solde_exp'])
df['diff_solde_dest'] = abs(df['old_solde_dest'] - df['new_solde_dest'])

# Sélection des features
df = df[['type', 'montant', 'diff_solde_orig', 'diff_solde_dest', 'fraude']]

# Normalisation des colonnes numériques
num_features = ['montant', 'diff_solde_orig', 'diff_solde_dest']
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# Encodage des variables catégorielles
cat_features = ['type']
encodage = OneHotEncoder(sparse_output=False)
encoded_cats = encodage.fit_transform(df[cat_features])  
encoded_df = pd.DataFrame(encoded_cats, columns=encodage.get_feature_names_out(cat_features))

# Concaténer et finaliser les données
df = pd.concat([df.drop(columns=cat_features), encoded_df], axis=1)

# Séparation des données
X = df.drop('fraude', axis=1)
y = df['fraude']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Données chargées en {time.time() - start_time:.2f} secondes.")

# 📌 Appliquer SMOTEENN
print("🔄 Application de SMOTEEN...")
start_time = time.time()

smoteenn = SMOTEENN()
X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train, y_train)

print(f"✅ SMOTEENN appliqué en {time.time() - start_time:.2f} secondes.")

# 📌 Entraîner le modèle
print("\n🔄 Entraînement du modèle KNN avec SMOTEENN...")
start_time = time.time()

model.fit(X_train_resampled, y_train_resampled)

print(f"✅ Modèle entraîné en {time.time() - start_time:.2f} secondes.")

# 📌 Prédictions sur l'ensemble de test
print("\n🔄 Prédictions sur l'ensemble de test...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilités pour la classe positive (fraude)

# Calcul des métriques
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Sauvegarder le modèle avec pickle
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(model, file)

