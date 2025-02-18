#Importation des bibliothèques
import pandas as pd
import openpyxl
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import os  

#Définition des chemins d'accès des fichiers Excel et CSV
xlsx_file = "00_ETL_excel.xlsx"  #Chemin du fichier Excel à convertir
csv_file = "ETL_csv.csv"  #Chemin du fichier CSV de sortie

try:
    #Vérification si le fichier CSV existe déjà
    if os.path.exists(csv_file):  #Si le fichier CSV existe déjà, ne rien faire
        print(f"Le fichier {csv_file} existe déjà. Aucun besoin de le recréer.")
    else:
        #Tentative d'ouverture du fichier Excel
        workbook = openpyxl.load_workbook(xlsx_file)  #Chargement du fichier Excel
        sheet = workbook.active  #Accéder à la première feuille active du fichier Excel

        #Créer le fichier CSV si il n'existe pas
        with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)  #Création de l'objet d'écriture CSV

            #Parcourir chaque ligne de la feuille Excel et écrire dans le fichier CSV
            for row in sheet.iter_rows(values_only=True):
                writer.writerow(row)  #Écrire chaque ligne dans le fichier CSV

        print(f"Conversion terminée : {csv_file}")  #Message de succès

except FileNotFoundError:
    #Si le fichier Excel n'est pas trouvé
    print(f"Erreur : Le fichier {xlsx_file} n'a pas été trouvé.")

except PermissionError:
    #Si le fichier CSV est ouvert ou inaccessible (problèmes de permission)
    print(f"Erreur : Impossible d'écrire dans le fichier {csv_file}. Vérifiez les permissions.")
    
except Exception as e:
    #Gérer d'autres types d'erreurs
    print(f"Une erreur inattendue s'est produite : {e}")

#Ouverture du fichier CSV pour l'affichage
df = pd.read_csv(csv_file)  #Lecture du fichier CSV
print(df.head())  #Afficher les premières lignes du fichier CSV