from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from sqlalchemy import create_engine, text

# Chargement du modèle
model = joblib.load('../models/xgb_oversampled_model.joblib')

# Connexion à la base (MySQL via phpMyAdmin)
DATABASE_URL = "mysql+pymysql://root:@localhost/doctolib"
engine = create_engine(DATABASE_URL)

required_cols = [ ... ]  # même liste que tu as
category_mappings = { ... }  # mêmes mappings que tu as

def encode_categories(df):
    for col, mapping in category_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)
    return df

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/pending_appointments")
async def get_pending_appointments():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM appointments WHERE status = 'pending'"))
        appointments = []
        for row in result.mappings():
            appointments.append({
                "id": row["id"],
                "features": {
                    'Scholarship': row["scholarship"],
                    'Hypertension': row["hypertension"],
                    'Diabetes': row["diabetes"],
                    'Alcoholism': row["alcoholism"],
                    'Disability': row["disability"],
                    'Days_Between_Scheduling_and_Appointment': row["days_between"],
                    'Hospital_Area': row["hospital_area"],
                    'Specialty': row["specialty"],
                    'Facility_Type': row["facility_type"],
                    'Distance_km': row["distance_km"],
                    'Type_of_Care': row["type_of_care"],
                    'Previously_Treated': row["previously_treated"],
                    'Age': row["age"],
                    'Social_Status': row["social_status"],
                    'SMS_Received': row["sms_received"],
                    'Weather_Conditions': row["weather_conditions"],
                    'Appointment_Time': row["appointment_time"],
                    'Gender': row["gender"],
                    'Consultations_Last_12_Months': row["consultations_last_12_months"],
                    'Waiting_Time_Minutes': row["waiting_time_minutes"],
                    'Hospital_Rating': row["hospital_rating"],
                    'Average_Fee': row["average_fee"],
                    'Number_days': row["number_days"]
                }
            })
    return {"appointments": appointments}

@app.post("/send_notification")
async def send_notification(request: Request):
    data = await request.json()
    appointment_id = data.get("appointment_id")
    message = "Vous devez confirmer à nouveau votre rendez-vous"
    
    # Insérer le message dans la base (notifications table ou colonne associée)
    with engine.connect() as conn:
        conn.execute(
            text("UPDATE appointments SET status = 'pending_confirmation', notification_message = :message WHERE id = :id"),
            {"message": message, "id": appointment_id}
        )
    
    return {"status": f"Notification enregistrée pour le rendez-vous ID {appointment_id}", "message": message}

@app.post("/predict")
async def predict(appointment: dict):
    df = pd.DataFrame([appointment])
    df = encode_categories(df)
    df = df[required_cols]
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    if not all(df.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        return {"error": "Les données contiennent encore des colonnes non numériques après encodage."}
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return {"prediction": int(prediction), "probability_annulation": float(probability)}
