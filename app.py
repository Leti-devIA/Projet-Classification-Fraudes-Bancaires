from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import sqlite3
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from werkzeug.security import generate_password_hash, check_password_hash

# Charger le modèle enregistré avec pickle
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialiser l'application Flask
app = Flask(__name__)

# Définir une clé secrète pour la gestion des sessions
app.secret_key = 'devIA'  # Remplacez par une clé unique et secrète

# Préparation du scaler et de l'encodeur (identiques à ceux utilisés pour l'entraînement)
scaler = StandardScaler()
encodage = OneHotEncoder(sparse_output=False)

# Fonction pour transformer les données comme lors de l'entraînement
def transform_data(df):
    """
    Transforme les données d'entrée pour les préparer à la prédiction.
    
    Args:
        df (DataFrame): Le DataFrame contenant les données à transformer.
    
    Returns:
        DataFrame: Le DataFrame transformé avec les caractéristiques normalisées et encodées.
    """
    # Transformation des noms des colonnes
    df.columns = ['id_trans', 'etape', 'type', 'montant', 'id_orig', 'old_solde_exp',
                  'new_solde_exp', 'id_dest', 'old_solde_dest', 'new_solde_dest', 'fraude']

    # Création de variables pour les différences de solde
    df['diff_solde_orig'] = abs(df['old_solde_exp'] - df['new_solde_exp'])
    df['diff_solde_dest'] = abs(df['old_solde_dest'] - df['new_solde_dest'])

    # Sélection des features pertinentes pour la prédiction
    df = df[['type', 'montant', 'diff_solde_orig', 'diff_solde_dest']]

    # Normalisation des colonnes numériques
    num_features = ['montant', 'diff_solde_orig', 'diff_solde_dest']
    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])

    # Encodage des variables catégorielles
    cat_features = ['type']
    encodage = OneHotEncoder(sparse_output=False)
    encoded_cats = encodage.fit_transform(df[cat_features])  
    encoded_df = pd.DataFrame(encoded_cats, columns=encodage.get_feature_names_out(cat_features))

    # Concaténer les données encodées avec les données numériques
    df = pd.concat([df.drop(columns=cat_features), encoded_df], axis=1)
    
    return df

# Chemin pour stocker le fichier CSV temporaire
TEMP_CSV_PATH = 'temp_data.csv'

name= []
# Exemple d'insertion des utilisateurs avec un mot de passe haché
# name = [("laeti@gmail.com", generate_password_hash("1234")), 
#         ("ps@gmail.com", generate_password_hash("1234")), 
#         ("merou@gmail.com", generate_password_hash("1234"))]

# Connexion à la base de données SQLite
with sqlite3.connect("database.db") as conn:
    cursor = conn.cursor()
    
    try:
        # Insérer chaque utilisateur avec le mot de passe haché
        for user_email, user_password in name:
            cursor.execute("""
                INSERT INTO users (id_user, password_user)
                VALUES (?, ?)""", (user_email, user_password))

        # Commit des changements dans la base de données après avoir inséré tous les utilisateurs
        conn.commit()
        print("Utilisateurs ajoutés avec succès.")
    
    except sqlite3.IntegrityError as e:
        # Si l'utilisateur existe déjà (conflit d'intégrité), ignorer l'erreur
        print(f"L'erreur d'intégrité : {e}")
        conn.rollback()  # Annuler la transaction en cas d'erreur

@app.route("/", methods=["GET", "POST"])
def accueil():
    """
    Route d'accueil pour la connexion des utilisateurs.
    
    Returns:
        str: Le rendu du template HTML pour la page d'accueil.
    """
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # Connexion à la BDD
        with sqlite3.connect("database.db") as conn:
            cursor = conn.cursor()

            # Vérifier si l'utilisateur existe dans la base
            cursor.execute("SELECT id_user, password_user FROM users WHERE id_user = ?", (email,))
            user = cursor.fetchone()
            
            if user and check_password_hash(user[1], password):
                # Si l'authentification réussit, enregistrer l'utilisateur dans la session
                session["user_id"] = user[0]
                return redirect(url_for("load"))
            else:
                # Si l'authentification échoue
                flash("Email ou mot de passe incorrect", "error")

    # Si la méthode est GET, afficher le formulaire de connexion
    return render_template("index.html")

@app.route("/load", methods=['GET', 'POST'])
def load():
    """
    Route pour le téléchargement de fichiers CSV.
    
    Returns:
        str: Le rendu du template HTML pour le téléchargement de fichiers.
    """
    if request.method == 'POST':
        # Vérifier si un fichier a été téléchargé
        if 'file' not in request.files:
            return "Aucun fichier téléchargé", 400
        
        file = request.files['file']
        
        # Vérifier si le fichier est un CSV
        if file.filename == '':
            return "Aucun fichier sélectionné", 400
        
        if file and file.filename.endswith('.csv'):
            # Sauvegarder le fichier CSV temporaire
            file.save(TEMP_CSV_PATH)
            return redirect(url_for('predict'))
        else:
            return "Le fichier doit être un fichier CSV", 400

    return render_template('load.html')

@app.route("/predict")
def predict():
    """
    Route pour effectuer des prédictions sur les données téléchargées.
    
    Returns:
        str: Le rendu du template HTML avec les résultats des prédictions.
    """
    try:
        # Lire le fichier CSV temporaire
        df = pd.read_csv(TEMP_CSV_PATH)
        
        nb_lignes = df.shape[0]
        
        # Appliquer la transformation des données (normalisation, encodage)
        df_transfo = transform_data(df)
        
        # Faire des prédictions
        predictions = model.predict(df_transfo)
        
        # Ajout des prédictions dans la colonne 'fraude_pred'
        df['fraude_pred'] = predictions 
        
        # Extraire les types uniques pour le filtre
        df_fraude = df[df['fraude_pred'] == 1]
        
        # Récupération du nombre de fraudes
        nombre_fraudes = df_fraude.shape[0]
        
        # Calculer le nombre de fraudes par type
        fraudes_par_type = df_fraude['type'].value_counts().to_dict()
        
        # Convertir le DataFrame en liste de dictionnaires
        data = df.to_dict(orient='records')
        
        # Extraire les types uniques pour le filtre
        types = df['type'].unique().tolist()
        
        # Insertion des résultats dans la table "predictions"
        with sqlite3.connect("database.db") as conn:
            cursor = conn.cursor()
            
            cursor.execute("""DELETE FROM predictions""")
            
            # Insérer chaque ligne dans la table predictions
            for index, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO predictions (id_trans, id_orig, id_dest, type, etape, montant, 
                        old_solde_exp, new_solde_exp, old_solde_dest, new_solde_dest, fraude_reel, fraude_pred)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (row['id_trans'], row['id_orig'], row['id_dest'], row['type'], row['etape'], row['montant'],
                     row['old_solde_exp'], row['new_solde_exp'], row['old_solde_dest'], row['new_solde_dest'], row['fraude'], row['fraude_pred'])
                )
            conn.commit()
        
        return render_template('predict.html', data=data, df=df, types=types, fraudes_par_type=fraudes_par_type, nombre_fraudes=nombre_fraudes)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Lancer l'application Flask en mode debug
    app.run(debug=True)