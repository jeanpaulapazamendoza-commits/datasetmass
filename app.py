import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from io import BytesIO

# ===========================================================
# CONFIGURACIÓN DE LA PÁGINA
# ===========================================================
st.set_page_config(
    page_title="Agrupación de Tiendas por Ubicación",
    page_icon="🏪",
    layout="wide"
)

# ===========================================================
# DATOS PERSONALES Y ENLACES (REQUERIDO POR EL ENUNCIADO)
# ===========================================================
URL_COLAB = "https://colab.research.google.com/drive/1HRFy03Da-KP6zSfyX6XSwvqeqqeaDUPP?usp=sharing"

# ===========================================================
# CARGA DEL DATASET Y MODELOS
# ===========================================================
@st.cache_data
def cargar_datos():
    df = pd.read_excel("Dataset.xlsx", sheet_name="df")
    return df

@st.cache_resource
def cargar_modelos():
    kmeans = joblib.load("modelos/modelo_kmeans.pkl")
    knn = joblib.load("modelos/modelo_knn.pkl")
    scaler = joblib.load("modelos/scaler.pkl")
    return kmeans, knn, scaler

# ===========================================================
# HEADER
# ===========================================================
st.title("🏪 Agrupación de Tiendas por Cercanía Geográfica")
st.markdown(f"""
**Alumno:** {NOMBRE_ALUMNO}
**Código ISIL:** {CODIGO_ISIL}
**Cuaderno de código (COLAB):** [Abrir en Google Colab]({URL_COLAB})
""")

st.markdown("---")

# ===========================================================
# DESCRIPCIÓN DEL PROBLEMA
# ===========================================================
with st.expander("📋 Descripción del problema y modelo", expanded=False):
    st.markdown("""
    ### Problema de negocio
    Una cadena de tiendas necesita optimizar las **rutas de despacho** a sus 54 sucursales en Lima
    (Villa El Salvador y San Juan de Miraflores). Agrupando las tiendas por cercanía geográfica
    se reducen costos de transporte y se mejora el tiempo de entrega.

    ### Modelos utilizados
    - **KMeans** (clustering no supervisado): forma los grupos de tiendas por distancia geográfica.
    - **K-Nearest Neighbors** (clasificador supervisado): predice el cluster de una tienda nueva.

    ### Cómo se usa esta app
    1. Selecciona en el panel lateral el número de grupos K que necesitas (o usa el sugerido).
    2. Observa el mapa interactivo con las tiendas coloreadas por cluster.
    3. Descarga el resultado como CSV.
    """)

# ===========================================================
# CARGAR DATOS
# ===========================================================
try:
    df = cargar_datos()
    kmeans_pretrained, knn_pretrained, scaler = cargar_modelos()
except FileNotFoundError as e:
    st.error(f"No se encontró el archivo: {e}")
    st.stop()

# ===========================================================
# SIDEBAR — CONTROLES DEL USUARIO
# ===========================================================
st.sidebar.header("⚙️ Configuración")

# Calcular K óptimo automáticamente
X = df[["latitud", "longitud"]].values
X_scaled = scaler.transform(X)

@st.cache_data
def calcular_k_optimo(_X_scaled):
    sils = {}
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(_X_scaled)
        sils[k] = silhouette_score(_X_scaled, labels)
    k_opt = max(sils, key=sils.get)
    return k_opt, sils

k_optimo, sil_dict = calcular_k_optimo(X_scaled)

st.sidebar.info(f"💡 **K óptimo sugerido:** {k_optimo}\n\n(Silhouette = {sil_dict[k_optimo]:.3f})")

# Slider para que el usuario elija K
k_usuario = st.sidebar.slider(
    "Selecciona el número de clusters (K):",
    min_value=2,
    max_value=10,
    value=k_optimo,
    step=1,
    help="K es el número de zonas de despacho en las que quieres agrupar las tiendas."
)

usar_optimo = st.sidebar.checkbox("Usar K óptimo automático", value=False)
K = k_optimo if usar_optimo else k_usuario

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Ver detalles")
mostrar_codo = st.sidebar.checkbox("Mostrar método del codo / silhouette", value=True)
mostrar_metricas = st.sidebar.checkbox("Mostrar métricas del modelo", value=True)

# ===========================================================
# ENTRENAR EL MODELO CON EL K SELECCIONADO
# ===========================================================
modelo = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = modelo.fit_predict(X_scaled)
df["cluster"] = df["cluster"].astype(str)  # para que plotly lo trate como categórico

# Centroides en escala original
centroides = scaler.inverse_transform(modelo.cluster_centers_)
df_centroides = pd.DataFrame(centroides, columns=["latitud", "longitud"])
df_centroides["cluster"] = [f"Centroide {i}" for i in range(K)]

# ===========================================================
# MÉTRICAS
# ===========================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Tiendas totales", len(df))
col2.metric("Clusters formados", K)
col3.metric("Silhouette Score", f"{silhouette_score(X_scaled, df['cluster']):.3f}")
col4.metric("Davies-Bouldin", f"{davies_bouldin_score(X_scaled, df['cluster']):.3f}")

st.markdown("---")

# ===========================================================
# MAPA INTERACTIVO
# ===========================================================
st.subheader(f"🗺️ Mapa interactivo — {K} grupos de tiendas")

fig_mapa = px.scatter_mapbox(
    df,
    lat="latitud",
    lon="longitud",
    color="cluster",
    hover_name="name_sucursal",
    hover_data={"codigo_sucursal": True, "distrito": True, "cluster": True,
                "latitud": False, "longitud": False},
    zoom=12,
    height=600,
    mapbox_style="open-street-map",
    color_discrete_sequence=px.colors.qualitative.Bold,
    title=f"Tiendas agrupadas en {K} zonas de despacho"
)

# Añadir centroides como X grandes
fig_mapa.add_trace(go.Scattermapbox(
    lat=df_centroides["latitud"],
    lon=df_centroides["longitud"],
    mode="markers",
    marker=dict(size=18, color="black", symbol="circle"),
    text=df_centroides["cluster"],
    name="Centroides",
    hoverinfo="text"
))

fig_mapa.update_layout(margin={"r":0, "t":40, "l":0, "b":0})
st.plotly_chart(fig_mapa, use_container_width=True)

# ===========================================================
# TABLA RESUMEN POR CLUSTER
# ===========================================================
st.subheader("📊 Resumen por cluster")
resumen = df.groupby("cluster").agg(
    cantidad_tiendas=("codigo_sucursal", "count"),
    lat_centro=("latitud", "mean"),
    lon_centro=("longitud", "mean")
).reset_index()
st.dataframe(resumen, use_container_width=True)

# ===========================================================
# GRÁFICAS DE MÉTODO DEL CODO Y SILHOUETTE
# ===========================================================
if mostrar_codo:
    st.subheader("📈 Selección del K óptimo")

    inercias = []
    silhouettes = []
    K_range = list(range(2, 11))
    for k in K_range:
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_tmp = km_tmp.fit_predict(X_scaled)
        inercias.append(km_tmp.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels_tmp))

    col_a, col_b = st.columns(2)

    with col_a:
        fig_codo = px.line(x=K_range, y=inercias, markers=True,
                           labels={"x": "K", "y": "Inercia (WCSS)"},
                           title="Método del Codo")
        fig_codo.add_vline(x=K, line_dash="dash", line_color="red",
                           annotation_text=f"K = {K} (actual)")
        st.plotly_chart(fig_codo, use_container_width=True)

    with col_b:
        fig_sil = px.line(x=K_range, y=silhouettes, markers=True,
                          labels={"x": "K", "y": "Silhouette Score"},
                          title="Silhouette Score por K")
        fig_sil.add_vline(x=K, line_dash="dash", line_color="red",
                          annotation_text=f"K = {K} (actual)")
        st.plotly_chart(fig_sil, use_container_width=True)

# ===========================================================
# MÉTRICAS DETALLADAS
# ===========================================================
if mostrar_metricas:
    st.subheader("📐 Interpretación de las métricas")
    st.markdown("""
    | Métrica | Valor actual | Interpretación |
    |---|---|---|
    | **Silhouette Score** | {:.3f} | Va de -1 a 1. Más cercano a 1 = clusters mejor separados. |
    | **Davies-Bouldin Index** | {:.3f} | Cuanto **más bajo**, mejor (clusters compactos y separados). |
    | **Inercia (WCSS)** | {:.4f} | Suma de distancias al centroide. Menor = clusters más densos. |
    """.format(
        silhouette_score(X_scaled, df["cluster"]),
        davies_bouldin_score(X_scaled, df["cluster"]),
        modelo.inertia_
    ))

# ===========================================================
# DETALLE DE TIENDAS Y DESCARGA
# ===========================================================
st.markdown("---")
st.subheader("📋 Detalle de tiendas asignadas a cada cluster")

df_descarga = df[["codigo_sucursal", "name_sucursal", "distrito",
                  "latitud", "longitud", "cluster"]].copy()
df_descarga = df_descarga.sort_values(["cluster", "name_sucursal"])

st.dataframe(df_descarga, use_container_width=True, height=400)

# Botón de descarga CSV
csv = df_descarga.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Descargar resultados (CSV)",
    data=csv,
    file_name=f"tiendas_agrupadas_K{K}.csv",
    mime="text/csv"
)

# ===========================================================
# PREDICTOR DE NUEVA TIENDA (MODELO 2: KNN)
# ===========================================================
st.markdown("---")
st.subheader("🆕 Predecir el cluster de una tienda NUEVA (modelo KNN)")
st.caption("Ingresa las coordenadas y el clasificador KNN entrenado te dirá a qué grupo pertenece.")

col_n1, col_n2, col_n3 = st.columns([1, 1, 1])
with col_n1:
    nueva_lat = st.number_input("Latitud", value=-12.18, format="%.6f")
with col_n2:
    nueva_lon = st.number_input("Longitud", value=-76.96, format="%.6f")
with col_n3:
    if st.button("🔍 Predecir cluster", use_container_width=True):
        # Re-entrenar KNN con los clusters actuales (porque el usuario pudo cambiar K)
        from sklearn.neighbors import KNeighborsClassifier
        knn_actual = KNeighborsClassifier(n_neighbors=3)
        knn_actual.fit(X_scaled, df["cluster"].astype(int))
        nueva_coord = scaler.transform([[nueva_lat, nueva_lon]])
        pred = knn_actual.predict(nueva_coord)[0]
        st.success(f"La nueva tienda pertenece al **Cluster {pred}**")

# ===========================================================
# FOOTER
# ===========================================================
st.markdown("---")
st.caption(f"""
Proyecto académico — Proceso de Aprendizaje 2 — ISIL
Modelos: KMeans (clustering) + KNN (clasificación supervisada) | Guardados en formato .pkl con joblib
[Ver cuaderno de código (Google Colab)]({URL_COLAB})
""")
