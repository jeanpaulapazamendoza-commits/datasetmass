"""
PA2 - Evaluación: Proceso de Aprendizaje 2
Agrupación de tiendas por cercanía geográfica (KMeans + KNN)
Autor:  Jean Paul Apaza Mendoza
Código: ISIL 76274929@mail.isil.pe
Curso:  Fundamentos de Machine Learning
"""
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
NOMBRE_ALUMNO = "Jean Paul Apaza mendoza"
CODIGO_ISIL = "ISIL 76274929@mail.isil.pe"
URL_COLAB = "https://colab.research.google.com/drive/1HRFy03Da-KP6zSfyX6XSwvqeqqeaDUPP?usp=sharing"

# ===========================================================
# FUNCIONES PARA CARGAR DATOS Y MODELOS (con cache)
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
# FUNCIÓN: GENERAR PLANTILLA EXCEL DE EJEMPLO
# ===========================================================
@st.cache_data
def generar_plantilla_excel():
    """Genera una plantilla Excel con las columnas correctas y filas de ejemplo."""
    plantilla = pd.DataFrame({
        "codigo_sucursal": [101, 102, 103],
        "name_sucursal": ["Tienda Centro Lima", "Tienda Miraflores", "Tienda San Isidro"],
        "distrito": ["Lima", "Miraflores", "San Isidro"],
        "latitud": [-12.046374, -12.119860, -12.097980],
        "longitud": [-77.042793, -77.029350, -77.036430]
    })
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        plantilla.to_excel(writer, index=False, sheet_name='df')
    return buffer.getvalue()

@st.cache_data
def generar_plantilla_csv():
    """Genera una plantilla CSV con las columnas correctas y filas de ejemplo."""
    plantilla = pd.DataFrame({
        "codigo_sucursal": [101, 102, 103],
        "name_sucursal": ["Tienda Centro Lima", "Tienda Miraflores", "Tienda San Isidro"],
        "distrito": ["Lima", "Miraflores", "San Isidro"],
        "latitud": [-12.046374, -12.119860, -12.097980],
        "longitud": [-77.042793, -77.029350, -77.036430]
    })
    return plantilla.to_csv(index=False).encode("utf-8")

# ===========================================================
# CARGAR MODELOS (siempre)
# ===========================================================
try:
    kmeans_pretrained, knn_pretrained, scaler = cargar_modelos()
except Exception as e:
    st.error(f"Error al cargar los modelos: {type(e).__name__}: {e}")
    st.stop()

# ===========================================================
# SIDEBAR — DESCARGAR PLANTILLA Y SUBIR ARCHIVO
# ===========================================================
st.sidebar.header("📂 Cargar tu propio dataset")

# Sub-sección: descargar plantilla
with st.sidebar.expander("📥 ¿No sabes cómo llenar los datos? Descarga la plantilla", expanded=False):
    st.caption("Descarga una plantilla con las columnas correctas y 3 filas de ejemplo. Edítala con tus propios datos y luego súbela abajo.")

    st.markdown("**Columnas requeridas:**")
    st.markdown("""
    | Columna | Tipo | Ejemplo |
    |---|---|---|
    | codigo_sucursal | número | 101 |
    | name_sucursal | texto | Tienda Centro Lima |
    | distrito | texto | Lima |
    | **latitud** ⭐ | decimal | -12.046374 |
    | **longitud** ⭐ | decimal | -77.042793 |
    """)
    st.caption("⭐ Solo latitud y longitud son obligatorias. Las otras son opcionales.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.download_button(
            label="📊 Excel",
            data=generar_plantilla_excel(),
            file_name="plantilla_tiendas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    with col_p2:
        st.download_button(
            label="📄 CSV",
            data=generar_plantilla_csv(),
            file_name="plantilla_tiendas.csv",
            mime="text/csv",
            use_container_width=True
        )

# Sub-sección: subir archivo
st.sidebar.caption("Sube un archivo Excel o CSV con tus tiendas:")

archivo_subido = st.sidebar.file_uploader(
    "Selecciona un archivo",
    type=["xlsx", "xls", "csv"],
    help="El archivo debe tener al menos las columnas 'latitud' y 'longitud'."
)

# Decidir qué dataset usar
if archivo_subido is not None:
    try:
        if archivo_subido.name.endswith(".csv"):
            df = pd.read_csv(archivo_subido)
        else:
            df = pd.read_excel(archivo_subido)

        columnas_requeridas = {"latitud", "longitud"}
        if not columnas_requeridas.issubset(df.columns):
            st.sidebar.error(f"⚠️ Faltan columnas. El archivo debe tener al menos: {columnas_requeridas}")
            st.stop()

        if "codigo_sucursal" not in df.columns:
            df["codigo_sucursal"] = range(1, len(df) + 1)
        if "name_sucursal" not in df.columns:
            df["name_sucursal"] = [f"Tienda {i}" for i in range(1, len(df) + 1)]
        if "distrito" not in df.columns:
            df["distrito"] = "Sin especificar"

        df = df.dropna(subset=["latitud", "longitud"]).reset_index(drop=True)

        if len(df) < 2:
            st.sidebar.error("⚠️ El archivo debe tener al menos 2 tiendas válidas.")
            st.stop()

        st.sidebar.success(f"✅ Archivo cargado: {len(df)} tiendas")
    except Exception as e:
        st.sidebar.error(f"Error al leer el archivo: {e}")
        st.stop()
else:
    try:
        df = cargar_datos()
        st.sidebar.info(f"ℹ️ Usando dataset por defecto ({len(df)} tiendas)")
    except Exception as e:
        st.sidebar.error(f"Error al cargar el dataset por defecto: {e}")
        st.stop()

st.sidebar.markdown("---")

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
    1. (Opcional) Descarga la plantilla y sube tu propio archivo Excel/CSV en el panel lateral.
    2. Selecciona el número de grupos K que necesitas (o usa el sugerido).
    3. Observa el mapa interactivo con las tiendas coloreadas por cluster.
    4. Descarga el resultado como CSV.
    """)

# ===========================================================
# SIDEBAR — CONTROLES DE CLUSTERING
# ===========================================================
st.sidebar.header("⚙️ Configuración del modelo")

X = df[["latitud", "longitud"]].values

if archivo_subido is not None:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = scaler.transform(X)

@st.cache_data
def calcular_k_optimo(_X_scaled, dataset_id):
    sils = {}
    max_k = min(11, len(_X_scaled))
    for k in range(2, max_k):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(_X_scaled)
        sils[k] = silhouette_score(_X_scaled, labels)
    k_opt = max(sils, key=sils.get)
    return k_opt, sils

dataset_id = "subido" if archivo_subido is not None else "default"
k_optimo, sil_dict = calcular_k_optimo(X_scaled, dataset_id)

st.sidebar.info(f"💡 **K óptimo sugerido:** {k_optimo}\n\n(Silhouette = {sil_dict[k_optimo]:.3f})")

max_k_slider = min(10, len(df) - 1)
k_usuario = st.sidebar.slider(
    "Selecciona el número de clusters (K):",
    min_value=2,
    max_value=max_k_slider,
    value=min(k_optimo, max_k_slider),
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
df["cluster"] = df["cluster"].astype(str)

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
    zoom=11 if archivo_subido is None else 4,
    height=600,
    mapbox_style="open-street-map",
    color_discrete_sequence=px.colors.qualitative.Bold,
    title=f"Tiendas agrupadas en {K} zonas de despacho"
)
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
    max_k = min(11, len(df))
    K_range = list(range(2, max_k))
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
