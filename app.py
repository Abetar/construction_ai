import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 📦 Cargar modelos
modelo_clas = joblib.load("modelo_clasificacion_realista.joblib")
modelo_reg = joblib.load("modelo_regresion_realista.joblib")

pre_clas = modelo_clas["preprocessor"]
mod_clas = modelo_clas["model"]
pre_reg = modelo_reg["preprocessor"]
mod_reg = modelo_reg["model"]

# 📋 Listas
meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
         'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
temporadas_por_mes = {
    'Enero': 'seca', 'Febrero': 'seca', 'Marzo': 'seca',
    'Abril': 'lluvias', 'Mayo': 'lluvias', 'Junio': 'ciclónica',
    'Julio': 'ciclónica', 'Agosto': 'ciclónica', 'Septiembre': 'ciclónica',
    'Octubre': 'lluvias', 'Noviembre': 'seca', 'Diciembre': 'seca'
}

# 🧠 Función auxiliar
def estimar_mes_retraso(mes_inicio, duracion_dias):
    idx = meses.index(mes_inicio)
    avance_meses = int(np.ceil(duracion_dias / 30))
    return meses[(idx + avance_meses) % 12]

def estimar_causa_probable(row):
    if row['lluvia_promedio_mm'] > 100:
        return "Clima"
    elif row['flujo_erogado_mdp'] < row['presupuesto_actualizado_mdp'] * 0.75:
        return "Financiero"
    elif row['contratos_por_asignar'] > 2:
        return "Contratos"
    elif row['trabajadores'] < 150:
        return "Mano de obra"
    else:
        return "Mixto"

# 🎨 Interfaz
st.set_page_config(page_title="Predicción de Retrasos", layout="centered")
st.title("🏗️ Predicción de Retrasos en Obras de Construcción")

with st.sidebar.form("formulario"):
    st.header("📋 Datos del Proyecto")

    tipo_obra = st.selectbox("Tipo de obra", [
        "Vivienda", "Escuela", "Hospital", "Puente", "Carretera", "Centro Comercial"
    ])
    region = st.selectbox("Región geográfica", [
        "Centro", "Occidente", "Noreste", "Sur", "Sureste", "Golfo"
    ])
    mes_inicio = st.selectbox("Mes de inicio", meses)
    temporada = temporadas_por_mes[mes_inicio]

    riesgo_sismico = st.selectbox("Riesgo sísmico", ["Bajo", "Medio", "Alto"])
    riesgo_inundacion = st.selectbox("Riesgo de inundación", ["Bajo", "Medio", "Alto"])

    duracion = st.number_input("Duración planeada (días)", 90, 540, 365)
    presupuesto_base = st.number_input("Presupuesto base (millones)", 5.0, 200.0, 50.0)
    presupuesto_act = st.number_input("Presupuesto actualizado (millones)", 5.0, 250.0, 55.0)
    flujo = st.number_input("Flujo erogado (millones)", 0.0, 250.0, 40.0)
    pendiente = presupuesto_act - flujo

    lluvia = st.number_input("Lluvia promedio estimada (mm)", 30.0, 200.0, 85.0)
    contratos = st.number_input("Contratos por asignar", 0, 10, 1)
    proveedores = st.number_input("Proveedores", 1, 30, 10)
    trabajadores = st.number_input("Trabajadores", 10, 1000, 300)
    m2 = st.number_input("Metros cuadrados de construcción", 100.0, 15000.0, 3000.0)

    submit = st.form_submit_button("🔍 Predecir")

# 🔮 Predicción
if submit:
    datos = pd.DataFrame([{
        "tipo_obra": tipo_obra,
        "region_geografica": region,
        "mes_inicio": mes_inicio,
        "temporada": temporada,
        "riesgo_sismico": riesgo_sismico,
        "riesgo_inundacion": riesgo_inundacion,
        "duracion_planeada_dias": duracion,
        "presupuesto_base_mdp": presupuesto_base,
        "presupuesto_actualizado_mdp": presupuesto_act,
        "flujo_erogado_mdp": flujo,
        "pendiente_en_caja_mdp": pendiente,
        "lluvia_promedio_mm": lluvia,
        "contratos_por_asignar": contratos,
        "proveedores": proveedores,
        "trabajadores": trabajadores,
        "m2_construccion": m2
    }])

    # Clasificación
    X_proc = pre_clas.transform(datos)
    prob = mod_clas.predict_proba(X_proc)[0][1]

    st.markdown("---")
    st.subheader("🔎 Resultados")

    if prob >= 0.7:
        st.error(f"🚨 Riesgo ALTO de retraso\n\n📊 Probabilidad: {prob * 100:.2f}%")
        mostrar_estimar = True
    elif prob >= 0.4:
        st.warning(f"⚠️ Riesgo MODERADO de retraso\n\n📊 Probabilidad: {prob * 100:.2f}%")
        mostrar_estimar = True
    else:
        st.success(f"✅ Riesgo BAJO de retraso\n\n📊 Probabilidad: {prob * 100:.2f}%")
        mostrar_estimar = False

    # Estimación adicional
    if mostrar_estimar:
        X_proc_r = pre_reg.transform(datos)
        dias_estimados = mod_reg.predict(X_proc_r)[0]
        mes_estimado = estimar_mes_retraso(mes_inicio, duracion)
        causa = estimar_causa_probable(datos.iloc[0])

        st.info(f"📆 Retraso estimado: **{dias_estimados:.0f} días**")
        st.info(f"🗓️ Impactaría alrededor de **{mes_estimado}**")
        st.info(f"❗ Causa probable: **{causa}**")
