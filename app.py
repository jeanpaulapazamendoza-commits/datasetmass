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
mostrar_metricas = st.sidebar.checkbox("Mostrar métricas del modelo", value=True)

# ===========================================================
# ENTRENAR EL MODELO
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
# MAPA INTERACTIVO PROFESIONAL
# ===========================================================
st.subheader(f"🗺️ Mapa interactivo — {K} zonas de despacho")

# Centro y zoom automáticos
center_lat = df["latitud"].mean()
center_lon = df["longitud"].mean()
lat_range = df["latitud"].max() - df["latitud"].min()
lon_range = df["longitud"].max() - df["longitud"].min()
max_range = max(lat_range, lon_range, 0.001)

if max_range < 0.05:
    zoom_calc = 13
elif max_range < 0.1:
    zoom_calc = 12
elif max_range < 0.3:
    zoom_calc = 11
elif max_range < 1:
    zoom_calc = 9
elif max_range < 5:
    zoom_calc = 6
elif max_range < 20:
    zoom_calc = 4
else:
    zoom_calc = 2

paletas = {
    "Vivid": px.colors.qualitative.Vivid,
    "Bold": px.colors.qualitative.Bold,
    "Pastel": px.colors.qualitative.Pastel,
    "Plotly": px.colors.qualitative.Plotly,
    "D3": px.colors.qualitative.D3,
    "Light24": px.colors.qualitative.Light24
}
colores = paletas[paleta_colores]

# Construir el mapa base
fig_mapa = go.Figure()

# 1) ZONAS DE COBERTURA (convex hull) — capa más al fondo
if mostrar_zonas:
    for i in range(K):
        cluster_data = df[df["cluster"] == str(i)]
        if len(cluster_data) >= 3:
            puntos = cluster_data[["longitud", "latitud"]].values
            try:
                hull = ConvexHull(puntos)
                hull_pts = puntos[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                color_cluster = colores[i % len(colores)]
                fig_mapa.add_trace(go.Scattermapbox(
                    lat=hull_pts[:, 1],
                    lon=hull_pts[:, 0],
                    mode="lines",
                    fill="toself",
                    fillcolor=hex_to_rgba(color_cluster, alpha=0.18),
                    line=dict(color=color_cluster, width=2),
                    name=f"Zona {i}",
                    hoverinfo="skip",
                    showlegend=False
                ))
            except Exception:
                pass

# 2) LÍNEAS GUÍA opcionales (capa intermedia)
if mostrar_lineas:
    for i in range(K):
        cluster_data = df[df["cluster"] == str(i)]
        cx = df_centroides["latitud"].iloc[i]
        cy = df_centroides["longitud"].iloc[i]
        color_cluster = colores[i % len(colores)]
        lats = []
        lons = []
        for _, row in cluster_data.iterrows():
            lats.extend([cx, row["latitud"], None])
            lons.extend([cy, row["longitud"], None])
        fig_mapa.add_trace(go.Scattermapbox(
            lat=lats, lon=lons,
            mode="lines",
            line=dict(color=hex_to_rgba(color_cluster, alpha=0.35), width=1),
            hoverinfo="skip",
            showlegend=False
        ))

# 3) TIENDAS (coloreadas por cluster)
for i in range(K):
    cluster_data = df[df["cluster"] == str(i)]
    color_cluster = colores[i % len(colores)]
    fig_mapa.add_trace(go.Scattermapbox(
        lat=cluster_data["latitud"],
        lon=cluster_data["longitud"],
        mode="markers",
        marker=dict(size=12, color=color_cluster, opacity=0.95),
        name=f"Cluster {i}",
        text=cluster_data["name_sucursal"],
        customdata=cluster_data[["codigo_sucursal", "distrito"]].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Código: %{customdata[0]}<br>"
            "Distrito: %{customdata[1]}<br>"
            "Lat: %{lat:.5f}<br>"
            "Lon: %{lon:.5f}<extra></extra>"
        )
    ))

# 4) CENTROIDES (capa al frente) — efecto anillo: capa negra + capa blanca + número
# Capa exterior negra (efecto anillo)
fig_mapa.add_trace(go.Scattermapbox(
    lat=df_centroides["latitud"],
    lon=df_centroides["longitud"],
    mode="markers",
    marker=dict(size=22, color="#1a1a1a", opacity=1.0),
    hoverinfo="skip",
    showlegend=False
))

# Capa intermedia blanca
fig_mapa.add_trace(go.Scattermapbox(
    lat=df_centroides["latitud"],
    lon=df_centroides["longitud"],
    mode="markers+text",
    marker=dict(size=18, color="white", opacity=1.0),
    text=[str(i) for i in range(K)],
    textfont=dict(size=11, color="#1a1a1a", family="Arial Black"),
    textposition="middle center",
    name="📍 Centroide",
    hovertext=[
        f"<b>Centroide del Cluster {i}</b><br>Lat: {df_centroides['latitud'].iloc[i]:.5f}<br>Lon: {df_centroides['longitud'].iloc[i]:.5f}"
        for i in range(K)
    ],
    hoverinfo="text"
))

fig_mapa.update_layout(
    mapbox=dict(
        style=estilo_mapa,
        center=dict(lat=center_lat, lon=center_lon),
        zoom=zoom_calc
    ),
    height=680,
    margin={"r": 0, "t": 10, "l": 0, "b": 0},
    legend=dict(
        title=dict(text="<b>Clusters</b>", font=dict(size=13, color="#222")),
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#888",
        borderwidth=1,
        font=dict(size=11, color="#222"),
        itemsizing="constant"
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial", bordercolor="#333")
)

st.plotly_chart(fig_mapa, use_container_width=True)

st.caption(
    "💡 Los **polígonos coloreados** son las zonas de cobertura (convex hull) de cada cluster. "
    "Los **círculos blancos numerados** son los centroides geográficos. "
    "Pasa el cursor sobre cualquier punto para ver el detalle."
)

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
# GRÁFICAS DEL CODO Y SILHOUETTE
# ===========================================================
if mostrar_codo:
    st.subheader("📈 Selección del K óptimo")
    inercias = []
    silhouettes = []
    max_k = min(MAX_K + 1, len(df))
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
# DETALLE Y DESCARGA
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
# PREDICTOR DE NUEVA TIENDA
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
