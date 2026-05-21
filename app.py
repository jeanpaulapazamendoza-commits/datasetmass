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
from scipy.spatial import ConvexHull
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
# DATOS PERSONALES Y ENLACES
# ===========================================================
NOMBRE_ALUMNO = "Jean Paul Apaza mendoza"
CODIGO_ISIL = "ISIL 76274929@mail.isil.pe"
URL_COLAB = "https://colab.research.google.com/drive/1HRFy03Da-KP6zSfyX6XSwvqeqqeaDUPP?usp=sharing"

MAX_K = 30

# ===========================================================
# FUNCIONES DE CARGA
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

@st.cache_data
def generar_plantilla_excel():
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
    plantilla = pd.DataFrame({
        "codigo_sucursal": [101, 102, 103],
        "name_sucursal": ["Tienda Centro Lima", "Tienda Miraflores", "Tienda San Isidro"],
        "distrito": ["Lima", "Miraflores", "San Isidro"],
        "latitud": [-12.046374, -12.119860, -12.097980],
        "longitud": [-77.042793, -77.029350, -77.036430]
    })
    return plantilla.to_csv(index=False).encode("utf-8")

def hex_to_rgba(hex_color, alpha=0.2):
    """Convierte color hex o nombre a rgba para usar en mapbox fill."""
    if hex_color.startswith("#"):
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
    if hex_color.startswith("rgb"):
        nums = hex_color.replace("rgb(", "").replace(")", "").split(",")
        r, g, b = [int(n.strip()) for n in nums[:3]]
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(100,100,100,{alpha})"

# ===========================================================
# CARGAR MODELOS
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
    st.caption("⭐ Solo latitud y longitud son obligatorias.")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.download_button("📊 Excel", data=generar_plantilla_excel(),
            file_name="plantilla_tiendas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
    with col_p2:
        st.download_button("📄 CSV", data=generar_plantilla_csv(),
            file_name="plantilla_tiendas.csv", mime="text/csv", use_container_width=True)

st.sidebar.caption("Sube un archivo Excel o CSV con tus tiendas:")
archivo_subido = st.sidebar.file_uploader(
    "Selecciona un archivo",
    type=["xlsx", "xls", "csv"],
    help="El archivo debe tener al menos las columnas 'latitud' y 'longitud'."
)

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
    Una cadena de tiendas necesita optimizar las **rutas de despacho** a sus sucursales en Lima.
    Agrupando las tiendas por cercanía geográfica se reducen costos de transporte y tiempos de entrega.
    ### Modelos utilizados
    - **KMeans** (clustering no supervisado): forma los grupos de tiendas por distancia geográfica.
    - **K-Nearest Neighbors** (clasificador supervisado): predice el cluster de una tienda nueva.
    ### Cómo se usa esta app
    1. (Opcional) Descarga la plantilla y sube tu propio archivo Excel/CSV.
    2. Selecciona el número de grupos K que necesitas.
    3. Observa el mapa con las **zonas de cobertura** coloreadas.
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
    max_k = min(MAX_K + 1, len(_X_scaled))
    for k in range(2, max_k):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(_X_scaled)
        sils[k] = silhouette_score(_X_scaled, labels)
    k_opt = max(sils, key=sils.get)
    return k_opt, sils

dataset_id = "subido" if archivo_subido is not None else "default"
k_optimo, sil_dict = calcular_k_optimo(X_scaled, dataset_id)

st.sidebar.info(f"💡 **K óptimo sugerido:** {k_optimo}\n\n(Silhouette = {sil_dict[k_optimo]:.3f})")

max_k_slider = min(MAX_K, len(df) - 1)
k_usuario = st.sidebar.slider(
    "Selecciona el número de clusters (K):",
    min_value=2,
    max_value=max_k_slider,
    value=min(k_optimo, max_k_slider),
    step=1,
    help=f"K es el número de zonas de despacho. Máximo permitido: {MAX_K}."
)

usar_optimo = st.sidebar.checkbox("Usar K óptimo automático", value=False)
K = k_optimo if usar_optimo else k_usuario

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Estilo del mapa")
estilo_mapa = st.sidebar.selectbox(
    "Selecciona el estilo:",
    options=["carto-positron", "open-street-map", "carto-darkmatter", "white-bg"],
    format_func=lambda x: {
        "carto-positron": "🌅 Claro (recomendado)",
        "open-street-map": "🗺️ OpenStreetMap",
        "carto-darkmatter": "🌃 Oscuro",
        "white-bg": "⬜ Blanco minimalista"
    }[x],
    index=0
)

paleta_colores = st.sidebar.selectbox(
    "Paleta de colores:",
    options=["Vivid", "Bold", "Pastel", "Plotly", "D3", "Light24"],
    index=0
)

mostrar_zonas = st.sidebar.checkbox("Mostrar zonas de cobertura", value=True,
    help="Dibuja un polígono coloreado alrededor de las tiendas de cada cluster.")

mostrar_lineas = st.sidebar.checkbox("Mostrar líneas guía a centroides", value=False,
    help="Conecta cada tienda con su centroide.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Ver detalles")
mostrar_codo = st.sidebar.checkbox("Mostrar método del codo / silhouette", value=True)
mostrar_metricas = st.sidebar.checkbox("Most
