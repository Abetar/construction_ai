import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from xgboost import plot_importance
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import locale
locale.setlocale(locale.LC_TIME, "Spanish_Mexico")
import plotly.graph_objects as go

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
    año_inicio = st.selectbox("Año de inicio", list(range(2023, 2027)))
    fecha_fin_programada = st.date_input("Fecha de fin programada")

    temporada = estimar_temporada(region, mes_inicio)  # Puedes ajustar a usar año si quieres más adelante

    riesgo_sismico = st.selectbox("Riesgo sísmico", ["Bajo", "Medio", "Alto"])
    riesgo_inundacion = st.selectbox("Riesgo de inundación", ["Bajo", "Medio", "Alto"])

    duracion = st.number_input("Duración planeada (días)", 90, 720, 365)
    m2 = st.number_input("Metros cuadrados de construcción", 100.0, 50000.0, 3000.0)

    presupuesto_base = st.number_input("Presupuesto base (millones)", 5.0, 500.0, 50.0)
    presupuesto_act = st.number_input("Presupuesto actualizado (millones)", 5.0, 600.0, 55.0)
    flujo = st.number_input("Flujo erogado (millones)", 0.0, 600.0, 40.0)
    pendiente = presupuesto_act - flujo
    restante = presupuesto_base - flujo

    contratos = st.number_input("Contratos por asignar", 0, 10, 1)
    trabajadores = st.number_input("Trabajadores", 10, 2000, 300)
    avance_prog = st.slider("Avance programado (%)", 0.0, 100.0, 60.0)
    avance_real = st.slider("Avance real actualizado (%)", 0.0, 100.0, 50.0)

    submit = st.form_submit_button("🔍 Predecir")


# 🔮 Predicción
if submit:
    # Calcular margen de tiempo desde fechas
    meses_dict = {
        "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
        "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
        "Septiembre": 9, "Octubre": 10, "Noviembre": 11, "Diciembre": 12
    }
    mes_inicio_num = meses_dict[mes_inicio]

    fecha_inicio_aprox = datetime(año_inicio, mes_inicio_num, 1)
    fecha_fin_dt = datetime.combine(fecha_fin_programada, datetime.min.time())
    margen_tiempo_dias = (fecha_fin_dt - fecha_inicio_aprox).days - duracion

    datos = pd.DataFrame([{
        "tipo_obra": tipo_obra,
        "region_geografica": region,
        "mes_inicio": mes_inicio,
        "año_inicio": año_inicio,
        "fecha_fin_programada": fecha_fin_programada,
        "riesgo_sismico": riesgo_sismico,
        "riesgo_inundacion": riesgo_inundacion,
        "temporada_climatica": temporada,
        "duracion_planeada_dias": duracion,
        "m2_construccion": m2,
        "presupuesto_base_mdp": presupuesto_base,
        "presupuesto_actualizado_mdp": presupuesto_act,
        "flujo_erogado_mdp": flujo,
        "pendiente_en_caja_mdp": pendiente,
        "presupuesto_restante_mdp": restante,
        "num_contratos_por_asignar": contratos,
        "num_trabajadores": trabajadores,
        "avance_programado_pct": avance_prog,
        "avance_real_pct": avance_real,
        "margen_tiempo_dias": margen_tiempo_dias
    }])

    # Clasificación
    X_proc = pre_clas.transform(datos)
    prob = mod_clas.predict_proba(X_proc)[0][1]

    st.markdown("---")
    st.subheader("🔎 Resultados")

    # Determinar nivel de riesgo
    if prob >= 0.7:
        riesgo_texto = "ALTO"
        color_fn = st.error
    elif prob >= 0.4:
        riesgo_texto = "MODERADO"
        color_fn = st.warning
    else:
        riesgo_texto = "BAJO"
        color_fn = st.success

    # Estimación de retraso si aplica
    mostrar_estimar = prob >= 0.4
    if mostrar_estimar:
        X_proc_r = pre_reg.transform(datos)
        dias_estimados = mod_reg.predict(X_proc_r)[0]

        # Ajustar si hay margen suficiente
        if margen_tiempo_dias >= 60 and dias_estimados < margen_tiempo_dias:
            riesgo_texto = "BAJO (por margen disponible)"
            color_fn = st.info

    color_fn(f"📊 Riesgo {riesgo_texto} de retraso\n\nProbabilidad: {prob * 100:.2f}%")

    # Estimación adicional
    if mostrar_estimar:
        fecha_final_ajustada = fecha_inicio_aprox + timedelta(days=duracion + dias_estimados)

        mes_retraso = fecha_final_ajustada.strftime("%B")
        año_retraso = fecha_final_ajustada.year

        causa_probable = (
            "Financiero" if flujo < presupuesto_act * 0.75 else
            "Ejecución lenta" if avance_real < avance_prog - 10 else
            "Clima" if temporada in ["lluvias", "ciclónica"] else
            "Multifactorial"
        )

        # Gráfico de línea de tiempo
        fig_tiempo = go.Figure()
        fig_tiempo.add_trace(go.Scatter(
            x=[fecha_inicio_aprox, fecha_final_ajustada],
            y=[1, 1],
            mode='lines+markers+text',
            line=dict(color='royalblue', width=4),
            marker=dict(size=10),
            text=["Inicio", "Fin estimado"],
            textposition="top center"
        ))
        fig_tiempo.update_layout(
            title="🕒 Línea de tiempo estimada del proyecto",
            xaxis_title="Fecha",
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_tiempo, use_container_width=True)

        # Mostrar info adicional
        st.info(f"📆 Retraso estimado: **{dias_estimados:.0f} días**")
        st.info(f"🗓️ El retraso impactaría en **{mes_retraso} de {año_retraso}**")
        st.info(f"📅 Nueva fecha estimada de término: **{fecha_final_ajustada.strftime('%d de %B de %Y')}**")
        st.info(f"❗ Causa probable: **{causa_probable}**")
