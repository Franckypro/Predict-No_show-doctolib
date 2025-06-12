import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sqlalchemy import create_engine, text
from sqlalchemy import create_engine

# --- Connexion à la base de données ---
engine = create_engine("mysql+pymysql://root:@localhost/doctolib")

# --- Chargement du modèle ---
model = joblib.load('../models/xgb_oversampled_model.joblib')

# --- Configuration de la page ---
st.set_page_config(page_title="Doctolib Annulation Prediction", layout="wide", page_icon="🩺")

# --- Ajout de style riche aux couleurs Doctolib et image de fond sur sidebar ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] > div:first-child {
            background-image: url('https://assets.entrepreneur.com/content/3x2/2000/1623253746-GettyImages-1273886962.jpg');
            background-size: cover;
            background-position: center;
            padding-top: 60px;
        }
        [data-testid="stSidebar"] .css-ng1t4o { 
            background-color: rgba(0, 123, 255, 0.8);
            border-radius: 12px;
            padding: 10px;
            color: white;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div {
            background-color: white;
            color: #007bff;
            border-radius: 8px;
        }
        .main { padding: 20px; }
        h1, h2, h3, h4 { color: #0069d9; text-align: center; animation: fadeIn 1.5s ease-in-out; }
        .stButton>button { background: linear-gradient(90deg, #0069d9, #2b9cd8); color: white; border-radius: 25px; padding: 12px 25px; font-size: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: none; }
        .stButton>button:hover { background: linear-gradient(90deg, #2b9cd8, #0069d9); }
        .highlight-box { background-color: #ffffff; border-left: 8px solid #0069d9; border-radius: 12px; padding: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin-bottom: 20px; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        .footer { text-align: center; margin-top: 50px; font-size: 12px; color: #0069d9; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Menu avec fond illustré ---
menu = st.sidebar.selectbox("Menu", ["Prédiction Temps Réel", "Classification sur CSV", "Système Automatique (Notifications)", "Tableaux de bord statistiques"])

# --- Affichage du logo ---
st.image("https://www.osteo-var.com/wp-content/uploads/2019/07/logo-doctolib.png", width=300)

# --- Section décorative ---
st.markdown("""
<div class="highlight-box" style="text-align:center; animation: fadeIn 2s ease-in-out; color: #0069d9;">
    <h3>🩺 Prévoyez mieux. Évitez les annulations. Améliorez votre planning.</h3>
    <p>Notre application vous aide à prédire et prévenir les absences, pour une meilleure organisation médicale.</p>
</div>
""", unsafe_allow_html=True)

st.title("Application Doctolib – Prédiction des Annulations de Rendez-vous")


required_cols = [
    'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Disability',
    'Days_Between_Scheduling_and_Appointment', 'Hospital_Area', 'Specialty',
    'Facility_Type', 'Distance_km', 'Type_of_Care', 'Previously_Treated', 'Age',
    'Social_Status', 'SMS_Received', 'Weather_Conditions', 'Appointment_Time',
    'Gender', 'Consultations_Last_12_Months', 'Waiting_Time_Minutes',
    'Hospital_Rating', 'Average_Fee', 'Number_days'
]

category_mappings = {
    'Hospital_Area': {'Pigalle': 13760, 'Bastille': 13887, 'Saint-Germain': 13846, 'Belleville': 13885, 'La Défense': 13835, 'Châtelet': 13768, 'Montparnasse': 13810},
    'Specialty': {'Pédiatrie': 15772, 'Gynécologie': 15785, 'Dermatologie': 15697, 'Cardiologie': 15892, 'Psychiatrie': 15771, 'Neurologie': 15778, 'Ophtalmologie': 15832},
    'Facility_Type': {'Conventionné': 0, 'Non conventionné': 1},
    'Type_of_Care': {'Vaccination': 21941, 'Urgence': 22224, 'Suivi': 22173, 'Bilan': 22018, 'Consultation': 22171},
    'Social_Status': {'Indépendant': 22195, 'Étudiant': 21999, 'Retraité': 22048, 'Sans emploi': 22007, 'Salarié': 22278},
    'Gender': {'Homme': 1, 'Femme': 0}
}

reverse_mappings = {col: {v: k for k, v in mapping.items()} for col, mapping in category_mappings.items()}

def encode_categories(df):
    for col, mapping in category_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)
    return df

def decode_categories(df):
    for col, mapping in reverse_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
    return df

def seconds_to_time(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def time_to_seconds(time_str):
    parts = time_str.split(':')
    if len(parts) == 2:
        h, m = map(int, parts)
        s = 0
    elif len(parts) == 3:
        h, m, s = map(int, parts)
    else:
        raise ValueError("Format d'heure invalide. Utilisez HH:MM ou HH:MM:SS")
    return h * 3600 + m * 60 + s
# Traductions françaises des champs
french_labels = {
    'Scholarship': "Bourse d'étude",
    'Hypertension': "Hypertension",
    'Diabetes': "Diabète",
    'Alcoholism': "Alcoolisme",
    'Disability': "Handicap",
    'Days_Between_Scheduling_and_Appointment': "Jours entre la prise et le rendez-vous",
    'Hospital_Area': "Zone hospitalière",
    'Specialty': "Spécialité",
    'Facility_Type': "Type d'établissement",
    'Distance_km': "Distance en km",
    'Type_of_Care': "Type de soin",
    'Previously_Treated': "Déjà traité",
    'Age': "Âge",
    'Social_Status': "Statut social",
    'SMS_Received': "SMS reçu",
    'Weather_Conditions': "Conditions météorologiques (0=Favorable, 1=Défavorable)",
    'Appointment_Time': "Heure du rendez-vous",
    'Gender': "Genre",
    'Consultations_Last_12_Months': "Consultations sur 12 mois",
    'Waiting_Time_Minutes': "Temps d'attente (min)",
    'Hospital_Rating': "Note de l'hôpital",
    'Average_Fee': "Frais moyens",
    'Number_days': "Nombre de jours"
}

# Ajout des champs français dans la prédiction
if menu == "Prédiction Temps Réel":
    st.subheader("Prédiction en Temps Réel")

    user_input = {}
    booking_date = st.date_input("Date de prise de rendez-vous")
    appointment_date = st.date_input("Date du rendez-vous")
    number_days = (appointment_date - booking_date).days
    user_input['Number_days'] = number_days

    for col in required_cols:
        if col == 'Number_days':
            continue
        label = french_labels.get(col, col)
        if col == 'Appointment_Time':
            time_str = st.text_input(f"{label} (HH:MM ou HH:MM:SS)", value="09:00")
            user_input[col] = time_to_seconds(time_str)
        elif col in ['Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Disability', 'SMS_Received', 'Previously_Treated']:
            user_input[col] = st.selectbox(f"{label} (Oui=1, Non=0)", [0, 1])
        elif col in category_mappings:
            user_input[col] = st.selectbox(label, list(category_mappings[col].keys()))
        else:
            user_input[col] = st.number_input(label, value=0)

    if st.button("Lancer la prédiction"):
        input_df = pd.DataFrame([user_input])
        input_df = encode_categories(input_df)
        input_df = input_df[required_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        prediction = model.predict(input_df)[0]
        probas = model.predict_proba(input_df)[0]
        st.success(f"Résultat : {'Annulation probable' if prediction == 1 else 'Présence probable'}")
        st.write(f"Probabilité d'annulation : {probas[1]*100:.2f}%")

elif menu == "Classification sur CSV":
    st.subheader("Classification en Masse (CSV)")
    uploaded_file = st.file_uploader("Téléverser un fichier CSV (avec colonnes exactes)", type=["csv"])
    if uploaded_file:
        st.write(" Fichier reçu côté Streamlit :")
        st.write(f"Nom du fichier : {uploaded_file.name}")
        st.write(f"Type de fichier : {uploaded_file.type}")
        st.write(f"Taille : {uploaded_file.size} octets")

        try:
            df_original = pd.read_csv(uploaded_file)
            df = df_original.copy()
            st.write(" Aperçu des premières lignes :")
            st.dataframe(df.head())
            st.write(" Colonnes détectées :", df.columns.tolist())

            if 'Appointment_Booking_Date' in df.columns and 'Appointment_Date' in df.columns:
                df['Appointment_Booking_Date'] = pd.to_datetime(df['Appointment_Booking_Date'])
                df['Appointment_Date'] = pd.to_datetime(df['Appointment_Date'])
                df['Number_days'] = (df['Appointment_Date'] - df['Appointment_Booking_Date']).dt.days
            if 'Appointment_Time' in df.columns:
                df['Appointment_Time'] = df['Appointment_Time'].apply(time_to_seconds)
            df_encoded = encode_categories(df)
            df_encoded = df_encoded[required_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

            predictions = model.predict(df_encoded)
            probas = model.predict_proba(df_encoded)[:,1]

            df_original['prediction'] = predictions
            df_original['proba_annulation'] = probas

            if 'Appointment_Time' in df_original.columns:
                df_original['Appointment_Time'] = df['Appointment_Time'].apply(seconds_to_time)
            df_original = decode_categories(df_original)

            st.success("✅ Prédictions terminées ! Voici les résultats :")

            def highlight_proba(val):
                return 'background-color: lightblue; color: black;'

            def highlight_prediction(val):
                color = 'background-color: red; color: white;' if val == 1 else 'background-color: green; color: white;'
                return color

            styled_df = df_original.style.applymap(highlight_proba, subset=['proba_annulation'])
            styled_df = styled_df.applymap(highlight_prediction, subset=['prediction'])

            st.dataframe(styled_df)

            csv_data = df_original.to_csv(index=False).encode('utf-8')
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_original.to_excel(writer, index=False, sheet_name='Predictions')
            excel_data = excel_buffer.getvalue()

            st.download_button("Télécharger en CSV", csv_data, file_name="predictions.csv", mime="text/csv")
            st.download_button("Télécharger en Excel", excel_data, file_name="predictions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f" Erreur lors du traitement du fichier : {e}")

elif menu == "Système Automatique (Notifications)":
    st.subheader("Système Automatique avec Notifications")
    st.write("⚠ Note : Cette fonction contacte une API locale. Assurez-vous que l'API est active et accepte les requêtes locales sans restriction (vérifiez les CORS et les permissions).")

    if st.button("Vérifier les rendez-vous à risque"):
        try:
            response = requests.get("http://localhost:8000/pending_appointments")
            response.raise_for_status()
            data = response.json()
            st.write(" Réponse reçue de l'API:", data)

            for appt in data['appointments']:
                input_df = pd.DataFrame([appt['features']])
                input_df = encode_categories(input_df)
                input_df = input_df[required_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
                prediction = model.predict(input_df)[0]
                if prediction == 1:
                    notif_response = requests.post("http://localhost:8000/send_notification", json={"appointment_id": appt['id']})
                    st.write(f"➡ Notification POST response: {notif_response.text}")

                    if notif_response.status_code == 200:
                        result = notif_response.json()
                        st.success(f"Notification envoyée pour le rendez-vous ID {appt['id']} - Statut: {result.get('status', 'OK')}")
                    else:
                        st.error(f"Erreur d'envoi pour ID {appt['id']} : {notif_response.status_code}, réponse : {notif_response.text}")
                else:
                    st.info(f"Aucun risque détecté pour le rendez-vous ID {appt['id']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la récupération ou de l'envoi : {e}")


# === DASHBOARD ===
elif menu == "Tableaux de bord statistiques":
    st.subheader("📊 Statistiques des rendez-vous depuis la base de données")
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM appointments"), conn)

        st.markdown("### Nombre de rendez-vous par spécialité")
        fig1, ax1 = plt.subplots()
        df['specialty'].value_counts().plot(kind='bar', color='#2b9cd8', ax=ax1)
        ax1.set_ylabel("Nombre de rendez-vous")
        ax1.set_xlabel("Spécialité")
        ax1.set_title("Répartition par spécialité")
        st.pyplot(fig1)

        st.markdown("### Statut des rendez-vous")
        fig2, ax2 = plt.subplots()
        df['status'].value_counts().plot.pie(autopct='%1.1f%%', colors=["#0069d9", "#28a745", "#dc3545"], ax=ax2)
        ax2.set_ylabel("")
        ax2.set_title("Répartition par statut")
        st.pyplot(fig2)

        st.markdown("### Répartition par zone hospitalière")
        fig3, ax3 = plt.subplots()
        sns.countplot(data=df, y="hospital_area", palette="Blues_r", order=df['hospital_area'].value_counts().index, ax=ax3)
        ax3.set_title("Zones hospitalières les plus utilisées")
        st.pyplot(fig3)

        st.markdown("### Âge des patients")
        fig4, ax4 = plt.subplots()
        sns.histplot(df['age'], bins=20, kde=True, color='#007bff', ax=ax4)
        ax4.set_title("Distribution des âges des patients")
        st.pyplot(fig4)

    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")

# --- Pied de page ---
st.markdown("""
<div class="footer">
    © 2025 Doctolib Predictor | Créé pour améliorer la santé numérique
</div>
""", unsafe_allow_html=True)
# --- Fin de l'application Streamlit ---