import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from xgboost import plot_importance
import numpy as np

# 📦 Cargar modelos v2
modelo_clas = joblib.load("modelo_clasificacion_v2.joblib")
modelo_reg = joblib.load("modelo_regresion_v2.joblib")

pre_clas = modelo_clas["preprocessor"]
mod_clas = modelo_clas["model"]
pre_reg = modelo_reg["preprocessor"]
mod_reg = modelo_reg["model"]

# Configuración de página
st.set_page_config(page_title="Predicción de Retrasos v2", layout="wide")
st.title("🏗️ Predicción de Retrasos en Obras de Construcción")

tabs = st.tabs(["📋 Formulario", "🧠 Explicación del modelo"])

with tabs[0]:
    # Aquí va TODO lo del formulario y predicción que ya tienes
    pass

with tabs[1]:
    st.markdown("### 🧠 ¿Qué variables pesan más en la predicción?")
    try:
        booster = mod_clas.get_booster()
        features = modelo_clas.get("features", [])

        # Asignar nombres reales al booster si están disponibles
        if features:
            booster.feature_names = features

        importancia = booster.get_score(importance_type='gain', fmap='')

        # Ordenar para mostrar en gráfica
        importancia_ordenada = sorted(importancia.items(), key=lambda x: x[1], reverse=True)
        variables = [x[0].replace("num__", "").replace("cat__", "") for x in importancia_ordenada]
        valores = [x[1] for x in importancia_ordenada]

        # Gráfico interactivo con Plotly
        fig = px.bar(
            x=valores, y=variables,
            orientation='h',
            labels={'x': 'Importancia (Gain)', 'y': 'Variable'},
            title='Importancia de variables en la predicción'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white' if st.get_option("theme.base") == "dark" else 'black'
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"No se pudo mostrar la importancia: {e}")
        
# Listas
meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
         'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
regiones = ['Noroeste', 'Noreste', 'Centro', 'Occidente', 'Sur', 'Sureste', 'Golfo']
temporadas_por_mes = {
    'Enero': 'seca', 'Febrero': 'seca', 'Marzo': 'seca',
    'Abril': 'seca', 'Mayo': 'seca', 'Junio': 'lluvias',
    'Julio': 'lluvias', 'Agosto': 'lluvias', 'Septiembre': 'lluvias',
    'Octubre': 'ciclónica', 'Noviembre': 'ciclónica', 'Diciembre': 'seca'
}

# Lógica para estimar temporada
def estimar_temporada(region, mes):
    if mes in ['Junio', 'Julio', 'Agosto', 'Septiembre']:
        return 'lluvias'
    elif mes in ['Octubre', 'Noviembre']:
        return 'ciclónica' if region in ['Sureste', 'Golfo'] else 'lluvias'
    else:
        return 'seca'

# Layout
with st.sidebar.form("formulario"):
    st.header("📋 Datos del Proyecto")

    tipo_obra = st.selectbox("Tipo de obra", [
        "Vivienda", "Escuela", "Hospital", "Puente", "Carretera", "Comercial"
    ])
    region = st.selectbox("Región geográfica", regiones)
    mes_inicio = st.selectbox("Mes de inicio", meses)
    temporada = estimar_temporada(region, mes_inicio)

    riesgo_sismico = st.selectbox("Riesgo sísmico", ["Bajo", "Medio", "Alto"])
    riesgo_inundacion = st.selectbox("Riesgo de inundación", ["Bajo", "Medio", "Alto"])

    duracion = st.number_input("Duración planeada (días)", 90, 720, 365)
    m2 = st.number_input("Metros cuadrados de construcción", 100.0, 50000.0, 3000.0)
    presupuesto_base = st.number_input("Presupuesto base (millones)", 5.0, 500.0, 50.0)
    presupuesto_act = st.number_input("Presupuesto actualizado (millones)", 5.0, 600.0, 55.0)
    flujo = st.number_input("Flujo erogado (millones)", 0.0, 600.0, 40.0)
    pendiente = presupuesto_act - flujo

    contratos = st.number_input("Contratos por asignar", 0, 10, 1)
    trabajadores = st.number_input("Trabajadores", 10, 2000, 300)
    avance_prog = st.slider("Avance programado (%)", 0.0, 100.0, 60.0)
    avance_real = st.slider("Avance real actualizado (%)", 0.0, 100.0, 50.0)

    submit = st.form_submit_button("🔍 Predecir")

# 🔮 Predicción
if submit:
    datos = pd.DataFrame([{
        "tipo_obra": tipo_obra,
        "region_geografica": region,
        "mes_inicio": mes_inicio,
        "riesgo_sismico": riesgo_sismico,
        "riesgo_inundacion": riesgo_inundacion,
        "temporada_climatica": temporada,
        "duracion_planeada_dias": duracion,
        "m2_construccion": m2,
        "presupuesto_base_mdp": presupuesto_base,
        "presupuesto_actualizado_mdp": presupuesto_act,
        "flujo_erogado_mdp": flujo,
        "pendiente_en_caja_mdp": pendiente,
        "num_contratos_por_asignar": contratos,
        "num_trabajadores": trabajadores,
        "avance_programado_pct": avance_prog,
        "avance_real_pct": avance_real
    }])

    # Clasificación
    X_proc = pre_clas.transform(datos)
    prob = mod_clas.predict_proba(X_proc)[0][1]

    st.markdown("---")
    st.subheader("🔎 Resultados")

    # Muestra textual
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
        mes_estimado = meses[(meses.index(mes_inicio) + int(np.ceil(duracion / 30))) % 12]
        causa_probable = (
            "Financiero" if flujo < presupuesto_act * 0.75 else
            "Ejecución lenta" if avance_real < avance_prog - 10 else
            "Clima" if temporada in ["lluvias", "ciclónica"] else
            "Multifactorial"
        )

        st.info(f"📆 Retraso estimado: **{dias_estimados:.0f} días**")
        st.info(f"🗓️ Se prevé que el impacto ocurra en **{mes_estimado}**")
        st.info(f"❗ Causa probable: **{causa_probable}**")
        
        # 🧠 Explicación contextual adicional
        factores_clave = []

        if avance_real < avance_prog - 10:
            factores_clave.append("avance real significativamente por debajo del programado")

        if flujo < presupuesto_act * 0.75:
            factores_clave.append("flujo financiero insuficiente respecto al presupuesto")

        if contratos > 2:
            factores_clave.append("múltiples contratos aún por asignar")

        if temporada in ["lluvias", "ciclónica"]:
            factores_clave.append("temporada climática adversa")

        if riesgo_inundacion == "Alto":
            factores_clave.append("riesgo de inundación elevado")

        if riesgo_sismico == "Alto":
            factores_clave.append("zona con alta actividad sísmica")

        # Generar resumen
        if factores_clave:
            explicacion = "Los factores que más podrían contribuir al retraso son: " + ", ".join(factores_clave) + "."
        else:
            explicacion = "No se detectaron factores críticos evidentes, pero el modelo estima riesgo por combinaciones sutiles de variables."

        st.markdown(f"🔍 **Análisis detallado del caso:** {explicacion}")