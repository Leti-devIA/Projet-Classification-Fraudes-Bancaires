import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import time
from tqdm import tqdm  # Bibliothèque pour la barre de progression
import pickle

# 📌 Définition du modèle et des hyperparamètres
model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    min_samples_split=2,
    max_features='log2'
)

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

# 📌 Appliquer SMOTE
print("🔄 Application de SMOTE...")
start_time = time.time()

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"✅ SMOTE appliqué en {time.time() - start_time:.2f} secondes.")

# 📌 Entraîner le modèle
print("\n🔄 Entraînement du modèle RandomForest avec SMOTE...")
start_time = time.time()

model.fit(X_train_resampled, y_train_resampled)

print(f"✅ Modèle entraîné en {time.time() - start_time:.2f} secondes.")

# 📌 Calcul des courbes d'apprentissage
print("\n🔄 Calcul des courbes d'apprentissage...")

train_sizes, train_scores, test_scores = learning_curve(
    model, X_train_resampled, y_train_resampled, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
)

# Moyenne et écart-type des scores d'entraînement et de test
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# 📊 Tracer la courbe d'apprentissage
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, color='blue', marker='o', label='Training score')
plt.plot(train_sizes, test_scores_mean, color='green', marker='s', label='Cross-validation score')

# Tracer les zones d'ombre pour l'écart-type
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, color='blue', alpha=0.2)
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, color='green', alpha=0.2)

plt.title("Learning Curve - Random Forest avec SMOTE")
plt.xlabel("Taille de l'ensemble d'entraînement")
plt.ylabel("Accuracy")
plt.legend(loc='best')
plt.grid(True)
plt.show()

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
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(model, file)

