"""
App de despliegue - Predicción de indicios de depresión en adolescentes.
Modelo: XGBoost entrenado en el notebook TrabajoFinalConIA2_0.ipynb
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Predicción Salud Mental Adolescente",
    page_icon="🧠",
    layout="centered"
)

# ============================================================
# CARGA DE ARTEFACTOS (caché para no recargar en cada interacción)
# ============================================================
@st.cache_resource
def load_artifacts():
    base = Path(__file__).parent
    return {
        'model':           joblib.load(base / 'xgb_model.pkl'),
        'scaler':          joblib.load(base / 'scaler.pkl'),
        'le_social':       joblib.load(base / 'le_social.pkl'),
        'feature_columns': joblib.load(base / 'feature_columns.pkl'),
        'numeric_cols':    joblib.load(base / 'numeric_cols.pkl'),
    }

art = load_artifacts()

# ============================================================
# UI
# ============================================================
st.title("🧠 Predicción de Indicios de Depresión")
st.caption("Modelo XGBoost · Trabajo Final IA · UPB")

st.markdown(
    "Esta herramienta usa hábitos digitales y de bienestar para estimar la "
    "**probabilidad de indicios de depresión** en un adolescente. "
    "⚠️ *No reemplaza un diagnóstico clínico — es solo un proyecto académico.*"
)

st.divider()

# ---------- INPUTS ----------
st.subheader("Datos del adolescente")

col1, col2 = st.columns(2)

with col1:
    daily_social_media_hours = st.slider(
        "Horas diarias en redes sociales", 0.0, 24.0, 4.0, 0.1)
    sleep_hours = st.slider(
        "Horas de sueño por día", 0.0, 18.0, 7.0, 0.1)
    screen_time_before_sleep = st.slider(
        "Tiempo de pantalla antes de dormir (h)", 0.0, 12.0, 1.5, 0.1)
    academic_performance = st.slider(
        "Rendimiento académico (0-5)", 0.0, 5.0, 3.5, 0.1)

with col2:
    physical_activity = st.slider(
        "Días de actividad física a la semana", 0.0, 7.0, 3.0, 1.0)
    addiction_level = st.slider(
        "Nivel de adicción a redes (0-10)", 0, 10, 5)
    social_interaction_level = st.selectbox(
        "Nivel de interacción social",
        options=['low', 'medium', 'high'],
        index=1,
        format_func=lambda x: {'low':'Bajo','medium':'Medio','high':'Alto'}[x]
    )

st.divider()

# ---------- PREDICCIÓN ----------
if st.button("🔮 Predecir", type="primary", use_container_width=True):
    # 1) Armar el registro en el mismo formato que vio el modelo
    record = {
        'daily_social_media_hours':  daily_social_media_hours,
        'sleep_hours':               sleep_hours,
        'screen_time_before_sleep':  screen_time_before_sleep,
        'academic_performance':      academic_performance,
        'physical_activity':         physical_activity,
        'social_interaction_level':  art['le_social'].transform([social_interaction_level])[0],
        'addiction_level':           addiction_level,
    }
    X_new = pd.DataFrame([record])[art['feature_columns']]  # mismo orden

    # 2) Escalar solo las numéricas (igual que en entrenamiento)
    X_new[art['numeric_cols']] = art['scaler'].transform(X_new[art['numeric_cols']])

    # 3) Predecir
    proba = art['model'].predict_proba(X_new)[0, 1]
    pred  = int(proba >= 0.5)

    # ---------- RESULTADO ----------
    st.subheader("Resultado")
    pct = proba * 100

    if pred == 1:
        st.error(f"⚠️ Indicios de depresión detectados (probabilidad: {pct:.1f}%)")
        st.markdown(
            "**Recomendación:** Consultar a un profesional de salud mental "
            "para una evaluación adecuada."
        )
    else:
        st.success(f"✅ Sin indicios claros de depresión (probabilidad: {pct:.1f}%)")
        st.markdown(
            "**Nota:** Esto no garantiza ausencia de problemas de salud mental. "
            "Si hay preocupaciones, hablar con un profesional siempre es válido."
        )

    # Barra visual de la probabilidad
    st.progress(np.clip(proba, 0.0, 1.0), text=f"Probabilidad de indicios: {pct:.1f}%")

    with st.expander("Ver detalle de la predicción"):
        st.json({
            "probabilidad_clase_1": round(float(proba), 4),
            "umbral_decision": 0.5,
            "prediccion_binaria": pred,
            "features_normalizadas": X_new.round(3).to_dict(orient='records')[0]
        })

st.divider()
st.caption(
    "Modelo entrenado con dataset *Teen_Mental_Health* (1148 registros tras limpieza). "
    "Limitación importante: la clase positiva es minoritaria (~3% del dataset), por lo "
    "que las predicciones de la clase 1 deben tomarse con cautela."
)
