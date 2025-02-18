import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import time
from tqdm import tqdm  # Bibliothèque pour la barre de progression

# Définition du modèle et des techniques d'équilibrage
model = RandomForestClassifier(random_state=42, n_jobs=-1)

sampling_techniques = {
    "SMOTE + TomekLinks": (SMOTE(random_state=42), TomekLinks())
}

# Charger les données
print("🔄 Chargement des données...")
start_time = time.time()

df = pd.read_csv("ETL_csv.csv")

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

# Appliquer SMOTE et TomekLinks avec barre de progression
print("🔄 Application de SMOTE et TomekLinks...")
start_time = time.time()

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

tomek_links = TomekLinks()
X_train_resampled, y_train_resampled = tomek_links.fit_resample(X_train_resampled, y_train_resampled)

print(f"✅ SMOTE + TomekLinks appliqués en {time.time() - start_time:.2f} secondes.")

# Définir les hyperparamètres pour RandomForest avec GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# GridSearchCV pour RandomForest
print("\n🔍 Démarrage de GridSearchCV pour RandomForest avec SMOTE + TomekLinks...")
start_time = time.time()

grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Affichage de la barre de progression pour l'entraînement
for _ in tqdm(range(1)):  # Simuler une barre de progression
    grid_search.fit(X_train_resampled, y_train_resampled)

print(f"✅ GridSearchCV terminé en {time.time() - start_time:.2f} secondes.")

# Meilleur modèle
best_model = grid_search.best_estimator_
print(f"✅ Meilleurs hyperparamètres : {grid_search.best_params_}")

# Prédictions sur l'ensemble de test
print("\n🔄 Prédictions sur l'ensemble de test...")
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probabilités pour la classe positive (fraude)

# Calcul des métriques
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# Création du graphique avec les pourcentages
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['Non Fraude', 'Fraude'], yticklabels=['Non Fraude', 'Fraude'])
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")
plt.title(f"Matrice de Confusion en % - RandomForest avec SMOTE + TomekLinks")
plt.savefig(f"matrice_confusion_RandomForest_SMOTE_TomekLinks_pourcentage.png")
plt.close()

# Courbe ROC
print("\n🔄 Calcul de la courbe ROC...")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Tracer la courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("Courbe ROC - RandomForest avec SMOTE + TomekLinks")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("roc_curve_RandomForest_SMOTE_TomekLinks.png")
plt.close()

