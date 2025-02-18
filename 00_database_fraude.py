#Importation des bibliothèques
import sqlite3
import pandas as pd
import os

#Nom du fichier de base de données
db_filename = 'database.db'

#Vérifier si le fichier de base de données existe déjà
if not os.path.exists(db_filename):

    #Si le fichier n'existe pas, le créer en se connectant à SQLite
    conn = sqlite3.connect(db_filename)
    print(f"Le fichier de base de données {db_filename} a été créé.")

    #Ouverture du CSV
    df = pd.read_csv("ETL_csv.csv")
    df.columns = ["id_trans", "etape", "type", "montant", "id_orig", "old_solde_exp", "new_solde_exp", 
                  "id_dest", "old_solde_dest", "new_solde_dest", "fraude"]

    #Connexion à la base de données SQLite
    cursor = conn.cursor()

    #Activer les clés étrangères
    cursor.execute("PRAGMA foreign_keys = ON;")

    #Création des tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS origine (
        id_orig TEXT PRIMARY KEY
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS destinataire (
        id_dest TEXT PRIMARY KEY
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactionne (
        id_orig TEXT,
        id_dest TEXT,
        id_trans INTEGER PRIMARY KEY,
        type TEXT,
        etape TEXT,
        montant REAL,
        old_solde_exp REAL,
        new_solde_exp REAL,
        old_solde_dest REAL,
        new_solde_dest REAL,
        fraude INTEGER,
        FOREIGN KEY(id_orig) REFERENCES origine(id_orig),
        FOREIGN KEY(id_dest) REFERENCES destinataire(id_dest)
    )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
        id_user TEXT NOT NULL UNIQUE,
        password_user TEXT NOT NULL)""" 
    )
            
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
        id_trans INTEGER PRIMARY KEY, 
        id_orig TEXT,
        id_dest TEXT,
        type TEXT,
        etape TEXT,
        montant REAL,
        old_solde_exp REAL,
        new_solde_exp REAL,
        old_solde_dest REAL,
        new_solde_dest REAL,
        fraude_reel INTEGER,
        fraude_pred INTEGER,
        FOREIGN KEY(id_orig) REFERENCES origine(id_orig),
        FOREIGN KEY(id_dest) REFERENCES destinataire(id_dest)
        )
        """
        )

    #Insérer les données par lot pour les tables origine et destinataire
    origine_data = [(id_orig,) for id_orig in df["id_orig"].drop_duplicates()]
    destinataire_data = [(id_dest,) for id_dest in df["id_dest"].drop_duplicates()]

    cursor.executemany("""
    INSERT OR REPLACE INTO origine (id_orig)
    VALUES (?)
    """, origine_data)

    cursor.executemany("""
    INSERT OR REPLACE INTO destinataire (id_dest)
    VALUES (?)
    """, destinataire_data)

    #Insérer les données par lot pour la table transactionne
    transactionne_data = df[[
        "id_orig", "id_dest", "id_trans", "type", "etape", "montant", 
        "old_solde_exp", "new_solde_exp", "old_solde_dest", "new_solde_dest", "fraude"
    ]].values.tolist()

    cursor.executemany("""
    INSERT OR REPLACE INTO transactionne (id_orig, id_dest, id_trans, type, etape, montant, 
                                          old_solde_exp, new_solde_exp, old_solde_dest, new_solde_dest, fraude)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, transactionne_data)

    #Sauvegarder les changements dans la base de données
    conn.commit()

    #Vérifier les données insérées
    try: 
        cursor.execute('SELECT * FROM transactionne LIMIT 5')
        rows = cursor.fetchall()
        for row in rows:
            print(row)
        print("Les données ont été insérées avec succès.")
    except Exception as e:
        print(f"Erreur : {e}")

    #Fermer la connexion à la base de données
    conn.close()

else: 
    print(f"{db_filename} déjà existant.")
