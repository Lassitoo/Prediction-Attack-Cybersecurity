import os
import pickle
import pandas as pd
import numpy as np
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponseRedirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .models import DataFileUpload
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Charger le modèle TensorFlow
MODEL_PATH = 'Apps/homeApp/Analyse/attack1.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")




# Initialiser le scaler et les encodeurs
scaler = StandardScaler()
label_encoders = {
    'protocol_type': LabelEncoder(),
    'service': LabelEncoder(),
    'flag': LabelEncoder()
}

# Fonction de prétraitement
def preprocess_data(data):
    """
    Prétraite les données :
    - Encode les colonnes catégoriques.
    - Met à l'échelle les colonnes numériques.
    """
    try:
        for column, encoder in label_encoders.items():
            if column in data.columns:
                if not hasattr(encoder, 'classes_'):
                    encoder.fit_transform(data[column])
                data[column] = encoder.transform(data[column])

        # Mise à l'échelle des colonnes numériques
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

        return data
    except Exception as e:
        raise ValueError(f"Erreur lors du prétraitement : {e}")
# Vues de base
def base(request):
    return render(request, 'homeApp/landing_page.html')


def upload_credit_data(request):
    return render(request, 'homeApp/upload_credit_data.html')


def prediction_button(request, id):
    return render(request, 'homeApp/fraud_detection.html', {'id': id})


def reports(request):
    all_data_files_objs = DataFileUpload.objects.all()
    return render(request, 'homeApp/reports.html', {'all_files': all_data_files_objs})


def enter_form_data_manually(request):
    # Liste des colonnes pour le formulaire
    columns = [
        "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files",
        "num_outbound_cmds", "is_host_login", "is_guest_login",
        "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "level"
    ]
    return render(request, 'homeApp/enter_form_data_manually.html', {'columns': columns})


def predict_data_manually(request):
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            input_data = {field: request.POST.get(field) for field in request.POST if field != 'csrfmiddlewaretoken'}
            print("Données reçues :", input_data)

            # Convertir les données en DataFrame pour le prétraitement
            data_df = pd.DataFrame([input_data])
            print("Données converties en DataFrame :", data_df)

            # Assurez-vous de convertir les valeurs en float pour les colonnes numériques
            for col in data_df.columns:
                if col not in ['protocol_type', 'service', 'flag']:  # Colonnes catégoriques
                    data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

            print("Données après conversion :", data_df)

            # Prétraitement des colonnes catégoriques
            categorical_columns = ['protocol_type', 'service', 'flag']
            for col in categorical_columns:
                if col in data_df.columns:
                    if col not in label_encoders:  # Vérifiez si un encodeur est défini
                        raise ValueError(f"Le label encoder pour {col} n'est pas défini.")
                    data_df[col] = label_encoders[col].fit_transform(data_df[col])

            print("Données après encodage :", data_df)
            print("hhhhhhhh", data_df.columns)

            # Mise à l'échelle des données
            data_preprocessed = scaler.fit_transform(data_df)
            print("Données prétraitées :", data_preprocessed)

            # Prédiction
            prediction = model.predict(data_preprocessed)
            result = 'attack' if prediction[0] > 0.5 else 'normal'

            context = {
                'status': result,
                'data': input_data,
                'proba': float(prediction[0]),  # Probabilité brute pour la classe prédite
            }
            return render(request, 'homeApp/predict_data_manually.html', context)

        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            messages.error(request, f"Erreur lors de la prédiction : {e}")
            return render(request, 'homeApp/predict_data_manually.html')

    return render(request, 'homeApp/predict_data_manually.html')


def add_files_single(request, id):
    return render(request,'homeApp/add_files_single.html', {'id': id})

from sklearn.metrics import accuracy_score


def predict_csv_single(request, id):
    if request.method == 'POST':
        try:
            # Charger le fichier CSV
            dataFrame = pd.read_csv(request.FILES['actual_file_name'])
            print("Données initiales du fichier CSV :", dataFrame.head())

            # Supprimer la colonne 'Unnamed: 0' si elle existe
            if 'Unnamed: 0' in dataFrame.columns:
                dataFrame = dataFrame.drop(columns=['Unnamed: 0'])
                print("Colonne 'Unnamed: 0' supprimée.")

            # Vérifier la présence de colonnes nécessaires
            expected_columns = ['protocol_type', 'service', 'flag']
            for col in expected_columns:
                if col not in dataFrame.columns:
                    raise ValueError(f"La colonne requise '{col}' est absente dans le fichier CSV.")

            # Extraire la première ligne pour la prédiction
            first_row = dataFrame.iloc[0:1]
            X = first_row.drop("attack", axis=1)

            # Prétraitement des colonnes catégoriques
            categorical_columns = ['protocol_type', 'service', 'flag']
            for col in categorical_columns:
                if col in X.columns:
                    if col not in label_encoders:
                        raise ValueError(f"L'encodeur pour la colonne '{col}' n'est pas défini.")

                    # Vérifier si l'encodeur est "ajusté" (fit)
                    if not hasattr(label_encoders[col], 'classes_'):
                        # Ajuster l'encodeur si ce n'est pas encore fait
                        known_classes = X[col].unique().tolist()
                        label_encoders[col].fit(known_classes)
                        print(f"L'encodeur pour '{col}' a été ajusté sur : {known_classes}")

                    # Vérifier si toutes les valeurs sont reconnues par l'encodeur
                    unseen_labels = set(X[col].unique()) - set(label_encoders[col].classes_)
                    if unseen_labels:
                        raise ValueError(f"Labels inconnus détectés dans '{col}': {unseen_labels}")

                    # Effectuer l'encodage
                    X[col] = label_encoders[col].fit_transform(X[col])

            print("Données après encodage :", X)

            # Mise à l'échelle des données
            X_scaled = scaler.fit_transform(X)
            print("Données mises à l'échelle :", X_scaled)

            # Prédiction pour la première ligne
            prediction = model.predict(X_scaled)
            predicted_class = 'Attack' if prediction[0] > 0.5 else 'Normal'

            # Ajouter la prédiction au contexte
            context = {
                'first_row': first_row.drop(columns=['attack']).to_dict(orient='records')[0],
                'prediction': predicted_class,
                'probability': float(prediction[0]),  # Probabilité brute
            }
            return render(request, 'homeApp/predict_csv_single.html', context)
        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            messages.error(request, f"Erreur : {e}")
            return redirect(f'/add_files_single/{id}')
    return render(request, 'homeApp/predict_csv_single.html')

def add_files_multi(request, id):
    return render(request,'homeApp/add_files_multi.html', {'id': id})

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA


def predict_csv_multi(request, id):
    if request.method == 'POST':
        try:
            # Charger le fichier CSV
            uploaded_file = request.FILES.get('actual_file_name')
            if not uploaded_file:
                raise ValueError("Veuillez télécharger un fichier CSV.")

            dataFrame = pd.read_csv(uploaded_file)
            print("Données initiales du fichier CSV :", dataFrame.head())

            # Supprimer la colonne 'Unnamed: 0' si elle existe
            if 'Unnamed: 0' in dataFrame.columns:
                dataFrame = dataFrame.drop(columns=['Unnamed: 0'])
                print("Colonne 'Unnamed: 0' supprimée.")

            # Vérifier la présence de colonnes nécessaires
            required_columns = ['protocol_type', 'service', 'flag', 'attack']
            for col in required_columns:
                if col not in dataFrame.columns:
                    raise ValueError(f"La colonne requise '{col}' est absente dans le fichier CSV.")

            # Récupérer le nombre de lignes à prédire
            num_rows = request.POST.get('num_rows', 10)  # Valeur par défaut : 10
            try:
                num_rows = int(num_rows)
                if num_rows < 1:
                    raise ValueError("Le nombre de lignes doit être au moins 1.")
            except ValueError:
                raise ValueError("Veuillez entrer un nombre valide pour le nombre de lignes.")

            dataFrame = dataFrame.head(num_rows)
            print(f"Prédiction limitée à {num_rows} lignes.")

            # Séparer les caractéristiques (X) et la cible (y)
            X = dataFrame.drop(columns=['attack'])
            y = dataFrame['attack']

            # Prétraitement des colonnes catégoriques
            categorical_columns = ['protocol_type', 'service', 'flag']
            for col in categorical_columns:
                if col in X.columns:
                    if col not in label_encoders:
                        # Créer un encodeur si inexistant
                        label_encoders[col] = LabelEncoder()

                    # Ajuster ou valider l'encodeur
                    if not hasattr(label_encoders[col], 'classes_'):
                        label_encoders[col].fit(X[col])
                        print(f"L'encodeur pour '{col}' a été ajusté sur : {list(label_encoders[col].classes_)}")

                    # Ajouter dynamiquement les nouvelles classes
                    known_classes = set(label_encoders[col].classes_)
                    new_classes = set(X[col].unique()) - known_classes
                    if new_classes:
                        print(f"Labels inconnus détectés dans '{col}': {new_classes}. Mise à jour de l'encodeur.")
                        label_encoders[col].classes_ = np.array(
                            list(known_classes | new_classes)
                        )

                    X[col] = X[col].apply(
                        lambda val: label_encoders[col].transform([val])[0] if val in label_encoders[
                            col].classes_ else -1
                    )

            print("Données après encodage :", X.head())

            # Mise à l'échelle des données
            X_scaled = scaler.fit_transform(X)
            print("Données mises à l'échelle :", X_scaled)

            # Faire des prédictions avec le modèle chargé
            predictions = model.predict(X_scaled)
            predicted_classes = (predictions > 0.5).astype(int)

            # Mapper les prédictions aux statuts
            result = ['Attack' if pred > 0.5 else 'Normal' for pred in predictions.flatten()]

            # Créer une liste de données combinées
            combined_data = []
            for record, status in zip(dataFrame.to_dict(orient='records'), result):
                combined_data.append({
                    'record': record,
                    'status': status
                })

            # Contexte pour le rendu
            context = {
                'id': id,
                'combined_data': combined_data,
                'num_rows': num_rows,
            }

            return render(request, 'homeApp/predict_csv_multi.html', context)

        except Exception as e:
            print(f"Erreur lors de la prédiction : {e}")
            messages.warning(request, f"Erreur : {e}")
            return redirect(f'/add_files_multi/{id}')

    return render(request, 'homeApp/predict_csv_multi.html')


def account_details(request):
    return render(request,'homeApp/account_details.html')
def change_password(request):
    return render(request,'homeApp/change_password.html')


from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelEncoder

def analysis(request, id):
    # Récupérer l'objet correspondant à l'ID
    obj = DataFileUpload.objects.get(id=id)

    # Charger les données du fichier CSV
    df = pd.read_csv(obj.actual_file.path)
    print("Colonnes présentes dans le fichier :", df.columns)

    # Vérifier les colonnes nécessaires
    if 'attack' not in df.columns:
        messages.error(request, "La colonne cible 'attack' est absente dans les données.")
        return redirect('/reports')

    # Standardiser les valeurs de la colonne 'attack' (gestion des fautes de casse)
    df['attack'] = df['attack'].str.strip().str.lower()

    # Encoder manuellement la colonne 'attack'
    mapping = {'attack': 1, 'normal': 0}
    if not set(df['attack']).issubset(mapping.keys()):
        messages.error(request, "La colonne 'attack' contient des valeurs non reconnues.")
        return redirect('/reports')

    df['attack'] = df['attack'].map(mapping)
    print("Colonnes encodées avec mapping :", df['attack'].unique())

    # Analyse des données
    empty_columns = len(df.columns[df.isnull().all()].tolist())  # Colonnes entièrement vides
    data_shape = df.shape  # Taille des données (lignes, colonnes)
    has_null = df.isnull().any().any()  # Vérifie si des valeurs nulles sont présentes

    # Calcul des pourcentages pour chaque type d'attaque
    total_rows = df.shape[0]
    percent_normal = 100 * (df[df['attack'] == 0].shape[0] / total_rows)  # Pourcentage normal
    percent_attack = 100 * (df[df['attack'] == 1].shape[0] / total_rows)  # Pourcentage attaque

    # Sous-ensembles de données
    normal_records = df[df['attack'] == 0]  # Données normales
    attack_records = df[df['attack'] == 1]  # Données d'attaque

    # Forme des sous-ensembles
    normal_shape = normal_records.shape
    attack_shape = attack_records.shape

    # Préparer les données pour le contexte de la vue
    context = {
        'data_shape': data_shape,  # Forme des données (lignes, colonnes)
        'unique_targets': list(mapping.keys()),  # Valeurs uniques avant encodage
        'percent_normal': round(percent_normal, 3),  # Pourcentage de données normales
        'percent_attack': round(percent_attack, 3),  # Pourcentage de données d'attaque
        'has_null': has_null,  # Présence de valeurs nulles
        'empty_columns': empty_columns,  # Nombre de colonnes vides
        'normal_shape': normal_shape,  # Forme des données normales
        'attack_shape': attack_shape,  # Forme des données d'attaque
    }

    # Renvoyer les résultats à la vue 'analysis.html'
    return render(request, 'homeApp/analysis.html', context)


def view_data(request,id):
    obj = DataFileUpload.objects.get(id=id)
    df = pd.read_csv(obj.actual_file.path)
    columns = df.columns.tolist()
    return render(request,'homeApp/view_data.html', {'id': id, 'columns': columns})
def delete_data(request,id):
    obj=DataFileUpload.objects.get(id=id)
    obj.delete()
    messages.success(request, "File Deleted succesfully",extra_tags = 'alert alert-success alert-dismissible show')
    return HttpResponseRedirect('/reports')


# Vue pour charger les données
from sklearn.exceptions import NotFittedError

def upload_data(request):
    if request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            data_file_name = request.POST.get('data_file_name')
            description = request.POST.get('description')
            actual_file = request.FILES['actual_file_name']

            # Sauvegarder le fichier
            fs = FileSystemStorage()
            file_path = fs.save(actual_file.name, actual_file)
            file_path = fs.path(file_path)

            # Charger les données
            data = pd.read_csv(file_path)

            # Supprimer les colonnes inutiles, si présentes
            if 'Unnamed: 0' in data.columns:
                data = data.drop(columns=['Unnamed: 0'])

            # Prétraitement des données
            data = preprocess_data(data)

            # Vérifier la présence de la colonne cible
            if 'attack' not in data.columns:
                raise ValueError("La colonne cible 'attack' est manquante dans les données.")

            X = data.drop(columns=['attack'])
            y = data['attack']

            # Encodage des étiquettes de la cible
            label_encoder = LabelEncoder()
            try:
                # Essayer de transformer les étiquettes
                y = label_encoder.fit_transform(y)
            except ValueError as ve:
                unseen_labels = set(y.unique()) - set(label_encoder.classes_)
                raise ValueError(f"Labels inattendus détectés dans 'attack' : {unseen_labels}")

            # Répartition des données
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Mise à l'échelle des données
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Sérialiser les données pour stockage
            x_test_serialized = pickle.dumps(X_test_scaled)
            y_test_serialized = pickle.dumps(y_test)

            # Sérialiser le modèle
            trained_model_data = pickle.dumps(model)

            # Enregistrer les données dans la base de données
            DataFileUpload.objects.create(
                file_name=data_file_name,
                actual_file=actual_file,
                description=description,
                # Uncomment the following lines if you want to save serialized data
                # trained_model_data=trained_model_data,
                # x_test_data=x_test_serialized,
                # y_test_data=y_test_serialized,
            )

            messages.success(request, "Fichier uploadé et enregistré avec succès.")
            return redirect('/reports')

        except ValueError as e:
            # Gérer les erreurs spécifiques liées aux étiquettes inattendues
            messages.error(request, f"Erreur lors du prétraitement : {e}")
            return redirect('/upload_credit_data')

        except Exception as e:
            # Log général des erreurs pour débogage
            messages.error(request, f"Erreur : {e}")
            return redirect('/upload_credit_data')

    return render(request, 'homeApp/upload_credit_data.html')

def retrieve_data_by_id(request, id):
    obj = DataFileUpload.objects.get(id=id)
    df = pd.read_csv(obj.actual_file.path)

    # Receive parameters from DataTables on the frontend
    draw = int(request.GET.get('draw', 1))
    start = int(request.GET.get('start', 0))
    length = int(request.GET.get('length', 10))

    # Paginate the data from the CSV using start and length
    paginated_df = df.iloc[start:start+length].reset_index()
    paginated_df['index'] = paginated_df['index'] + 1 + start

    # Convert the paginated data to a list of lists
    data = paginated_df.values.tolist()

    # Return a JSON response suitable for DataTables
    return JsonResponse({
        'draw': draw,
        'recordsTotal': len(df),
        'recordsFiltered': len(df),  # In case you add server-side filtering later on
        'data': data,
    })

def userLogout(request):
    try:
      del request.session['username']
    except:
      pass
    logout(request)
    return HttpResponseRedirect('/')


def login2(request):
    data = {}
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        print(user)
        if user:
            login(request, user)
            return HttpResponseRedirect('/')

        else:
            data['error'] = "Username or Password is incorrect"
            res = render(request, 'homeApp/login.html', data)
            return res
    else:
        return render(request, 'homeApp/login.html', data)


def about(request):
    return render(request,'homeApp/about.html')

def dashboard(request):
    return render(request,'homeApp/dashboard.html')


from django.http import JsonResponse
from sklearn.preprocessing import StandardScaler, LabelEncoder


import requests
import pandas as pd
from django.shortcuts import render
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Importez vos encodeurs et modèles

import requests
from urllib.parse import urlparse

def extract_attack_features(url):
    # Initialisation des colonnes avec des valeurs par défaut
    features = {
        'duration': 0, 'protocol_type': None, 'service': None, 'flag': None,
        'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0,
        'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 0,
        'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0,
        'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
        'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0,
        'is_guest_login': 0, 'count': 0, 'srv_count': 0, 'serror_rate': 0,
        'srv_serror_rate': 0, 'rerror_rate': 0, 'srv_rerror_rate': 0,
        'same_srv_rate': 0, 'diff_srv_rate': 0, 'srv_diff_host_rate': 0,
        'dst_host_count': 0, 'dst_host_srv_count': 0, 'dst_host_same_srv_rate': 0,
        'dst_host_diff_srv_rate': 0, 'dst_host_same_src_port_rate': 0,
        'dst_host_srv_diff_host_rate': 0, 'dst_host_serror_rate': 0,
        'dst_host_srv_serror_rate': 0, 'dst_host_rerror_rate': 0,
        'dst_host_srv_rerror_rate': 0, 'level': 0
    }

    try:
        # Parse the URL
        parsed_url = urlparse(url)

        # Remplir les colonnes pertinentes
        features['protocol_type'] = parsed_url.scheme  # http, https
        features['service'] = parsed_url.netloc.split('.')[-1]  # Domaine comme 'com'
        features['flag'] = 'SF'  # Exemple par défaut, à ajuster si pertinent
        features['src_bytes'] = len(url)  # Longueur de l'URL comme approximation
        features['dst_bytes'] = len(parsed_url.netloc)  # Longueur du domaine comme approximation

        # Simulation pour d'autres colonnes
        features['duration'] = 1  # Statique pour l'instant
        features['logged_in'] = 1 if parsed_url.scheme == 'https' else 0  # HTTPS comme indicateur

    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques pour l'URL {url} : {e}")

    return features



from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .models import Prediction
import pandas as pd

# Vue pour prédiction en temps réel
def predict_realtime(request):
    if request.method == 'POST':
        try:
            # Récupérer l'URL
            url = request.POST.get('url')
            if not url:
                raise ValueError("Veuillez entrer une URL valide.")

            # Extraire les caractéristiques de l'URL
            features = extract_attack_features(url)
            print("Caractéristiques extraites :", features)

            # Convertir les caractéristiques en DataFrame
            data_df = pd.DataFrame([features])

            # Liste des colonnes catégoriques
            categorical_columns = ['protocol_type', 'service', 'flag']

            # Encodage des colonnes catégoriques
            for col in categorical_columns:
                if col in data_df.columns:
                    # Si un LabelEncoder existe pour cette colonne, utilisez-le
                    if col in label_encoders:
                        data_df[col] = label_encoders[col].fit_transform(data_df[col])
                    else:
                        raise ValueError(f"L'encodeur pour '{col}' n'est pas défini.")

            # Mise à l'échelle des données
            data_scaled = scaler.fit_transform(data_df)

            # Faire la prédiction
            prediction = model.predict(data_scaled)
            status = "Attack" if prediction[0] > 0.5 else "Normal"

            # Si le protocole est HTTPS et le statut est "Attack", changer en "Normal"
            if features.get('protocol_type', '').lower() == 'https' and status == "Attack":
                status = "Normal"
                recommendation = "Le protocole HTTPS garantit une sécurité suffisante."
            else:
                # Générer une recommandation
                recommendation = (
                    "Renforcez les mesures de sécurité." if status == "Attack" else "Aucune action nécessaire."
                )

            # Sauvegarder la prédiction dans la base de données
            Prediction.objects.create(
                url=url,
                prediction_result=status,
                extracted_features=features,
                recommendation=recommendation
            )

            # Rediriger vers la liste des prédictions
            return redirect('historique_predictions')

        except Exception as e:
            print(f"Erreur lors de la prédiction en temps réel : {e}")
            return render(request, 'homeApp/predict_realtime.html', {"error": str(e)})

    return render(request, 'homeApp/predict_realtime.html')

# Vue pour liste des prédictions
def prediction_list(request):
    predictions = Prediction.objects.all().order_by('-date')
    return render(request, 'homeApp/prediction_list.html', {"predictions": predictions})

# Vue pour suppression
def delete_prediction(request, id):
    if request.method == 'POST':
        prediction = get_object_or_404(Prediction, id=id)
        prediction.delete()
        messages.success(request, "La prédiction a été supprimée.")
        return redirect('historique_predictions')

def prediction_detail(request, id):
    # Récupérer la prédiction spécifique
    prediction = get_object_or_404(Prediction, id=id)
    context = {
        "prediction": prediction
    }
    return render(request, 'homeApp/prediction_detail.html', context)

import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from django.shortcuts import render
from .models import Prediction

def analyze_predictions(request):
    predictions = Prediction.objects.all()

    # Préparation des données pour l'analyse
    data = {
        "Date": [p.date for p in predictions],
        "Statut": [p.prediction_result for p in predictions],
        "Recommandation": [p.recommendation for p in predictions],
        "Protocol Type": [p.extracted_features.get("protocol_type", "Unknown") for p in predictions],
        "Attack": [p.prediction_result == "Attack" for p in predictions],
    }

    df = pd.DataFrame(data)

    # Graphique 1 : Distribution des Statuts
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Statut", palette="coolwarm")
    plt.title("Distribution des Prédictions (Attaque vs Normal)")
    plt.xlabel("Statut de la Prédiction")
    plt.ylabel("Nombre")
    plt.tight_layout()
    buffer1 = BytesIO()
    plt.savefig(buffer1, format="png")
    buffer1.seek(0)
    image_base64_1 = base64.b64encode(buffer1.getvalue()).decode("utf-8")
    buffer1.close()

    # Graphique 2 : Attaques par Protocol Type
    plt.figure(figsize=(16, 4))
    sns.countplot(x="Attack", data=df, hue="Protocol Type", palette="viridis")
    plt.xticks(rotation=45)
    plt.title("Attack Counts over Protocol Types", fontdict={"fontsize": 16})
    plt.tight_layout()
    buffer2 = BytesIO()
    plt.savefig(buffer2, format="png")
    buffer2.seek(0)
    image_base64_2 = base64.b64encode(buffer2.getvalue()).decode("utf-8")
    buffer2.close()

    # Contexte pour les graphiques
    context = {
        "image_base64_1": image_base64_1,
        "image_base64_2": image_base64_2,
        "total_predictions": len(predictions),
        "attack_count": df["Statut"].value_counts().get("Attack", 0),
        "normal_count": df["Statut"].value_counts().get("Normal", 0),
    }

    return render(request, "homeApp/analyze_predictions.html", context)
