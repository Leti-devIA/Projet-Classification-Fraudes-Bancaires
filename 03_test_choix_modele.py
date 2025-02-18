import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTEN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier



# ------------------------------------------- FONCTIONS ------------------------------------------- #

# Fonction pour appliquer différentes techniques de rééquilibrage des classes
def fonction_sampling(X_train, y_train, sampling_technique='None'):
    """
    Applique différentes techniques de rééchantillonnage pour équilibrer les classes.
    
    :param X_train: Données d'entraînement (features)
    :param y_train: Labels des données d'entraînement
    :param sampling_technique: Technique d'échantillonnage à appliquer
    :return: Données rééchantillonnées (X_train_res, y_train_res)
    
    """
    if sampling_technique == 'SMOTE':
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    elif sampling_technique == 'NearMiss':
        nearmiss = NearMiss()
        X_train_res, y_train_res = nearmiss.fit_resample(X_train, y_train)
    elif sampling_technique == 'SMOTEN':
        smoten = SMOTEN()
        X_train_res, y_train_res = smoten.fit_resample(X_train, y_train)
    elif sampling_technique == "SMOTETomek":
        smotetomek = SMOTETomek()
        X_train_res, y_train_res = smotetomek.fit_resample(X_train, y_train)
    elif sampling_technique == 'SMOTEENN':
        smoteenn = SMOTEENN()
        X_train_res, y_train_res = smoteenn.fit_resample(X_train, y_train)
    else:
        # Aucun rééchantillonnage appliqué
        X_train_res, y_train_res = X_train, y_train
        
    return X_train_res, y_train_res



# Fonction pour tester différents modèles
def tester_modele(modele, X_train, y_train, X_test, y_test, sampling):
    """
    Entraîne et évalue un modèle avec une technique d'échantillonnage donnée.
    
    :param modele: Modèle de machine learning à tester
    :param X_train: Données d'entraînement
    :param y_train: Labels d'entraînement
    :param X_test: Données de test
    :param y_test: Labels de test
    :param sampling: Technique d'échantillonnage utilisée
    
    """
    # Impressions pour savoir où en est le test 
    print(f"\n🔍 Test du modèle : {modele.__class__.__name__} avec {sampling}")
    modele.fit(X_train, y_train)  # Entraînement du modèle
    y_pred = modele.predict(X_test)  # Prédictions

    # Affichage des métriques de performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Affichage de la matrice de confusion
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=['Non Fraude', 'Fraude'], 
                yticklabels=['Non Fraude', 'Fraude'])
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies valeurs")
    plt.title(f"Matrice de Confusion - {modele.__class__.__name__} ({sampling})")
    plt.show()


# ------------------------------------------- FICHIER ------------------------------------------- #

# Ouverture du fichier 
df = pd.read_csv("ETL_csv.csv")

# ------------------------------------------- NOUVELLES COLONNES -------------------------------------------#

# Transformation des noms des colonnes
df.columns = ['id_trans', 'etape', 'type', 'montant', 'id_orig', 'old_solde_exp',
              'new_solde_exp', 'id_dest', 'old_solde_dest', 'new_solde_dest',
              'fraude']

# Création de nouvelles features 
# Création de la colonne qui fait la différence sur le solde de l'expéditeur
df['diff_solde_orig'] = abs(df['old_solde_exp'] - df['new_solde_exp'])

# Création de la colonne qui fait la différence sur le solde du destinataire
df['diff_solde_dest'] = abs(df['old_solde_dest'] - df['new_solde_dest'])

# Ne garder que les colonnes qu'on juge utiles
df = df[['type', 'montant', 'diff_solde_orig', 'diff_solde_dest', 'fraude']]

print(df.head())

# ------------------------------------------- NORMALISATION ------------------------------------------- #

# Définir les colonnes numériques et catégorielles
num_features = ['montant', 'diff_solde_orig', 'diff_solde_dest']  # Ajoutez ici vos colonnes numériques
cat_features = ['type']  # Ajoutez ici vos colonnes catégorielles

# Sur données numériques
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

#Sur données catégorielles
encodage = OneHotEncoder(sparse_output=False)  # `sparse=False` pour obtenir un tableau dense
encoded_type = encodage.fit_transform(df[['type']])

# Convertir l'array en DataFrame et ajouter les colonnes encodées à df
df_encoded = pd.DataFrame(encoded_type, columns=encodage.get_feature_names_out(['type']))
df = pd.concat([df, df_encoded], axis=1)

# Suppression de la colonne 'type' initiale
df = df.drop(['type'], axis=1)

# Encodage de la colonne 'type' en valeurs numériques (uniquement utilisé pour le RandomForest)
# label_encoder = LabelEncoder()
# df['type'] = label_encoder.fit_transform(df['type'])

print(df.head())


# ------------------------------------------- TESTS ------------------------------------------- #

# Diviser les données d'entrainement
X = df.drop('fraude', axis=1)
y = df['fraude']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Liste des modèles à tester
modeles = [
    # LogisticRegression(),
    # XGBClassifier(),
    # GaussianNB(),
    # KNeighborsClassifier(), 
    # RandomForestClassifier(),
    # LGBMClassifier(),
    GradientBoostingClassifier()

]

# Liste des techniques de rééchantillonnage
liste_techniques = ["SMOTE", "SMOTETomek"]

# Boucle pour tester chaque modèle avec chaque technique d'échantillonnage
for sampling in liste_techniques:
    X_train_res, y_train_res = fonction_sampling(X_train, y_train, sampling)
    for modele in modeles:
        tester_modele(modele, X_train_res, y_train_res, X_test, y_test, sampling)
