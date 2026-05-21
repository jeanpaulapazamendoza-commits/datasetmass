"""
PA2 - Evaluación: Proceso de Aprendizaje 2
Agrupación de tiendas por cercanía geográfica (KMeans + KNN) + Ruteo óptimo
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
import openrouteservice as ors
from openrouteservice import optimization, convert

st.set_page_config(
    page_title="Agrupación de Tiendas y Ruteo Óptimo",
    page_icon="🏪",
    layout="wide"
)

NOMBRE_ALUMNO = "Jean Paul Apaza mendoza"
CODIGO_ISIL = "ISIL 76274929@mail.isil.pe"
URL_COLAB = "https://colab.research.google.com/drive/1HRFy03Da-KP6zSfyX6XSwvqeqqeaDUPP?usp=sharing"
MAX_K = 30

# ===========================================================
# FUNCIONES
# ===========================================================
@st.cache_data
def cargar_datos():
    return pd.read_excel("Dataset.xlsx", sheet_name="df")

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

def calcular_ruta_ors(cd_coord, tiendas_coords, api_key, cerrada=True):
    client = ors.Client(key=api_key)
    jobs = [optimization.Job(id=int(idx), location=[float(lon), float(lat)])
            for (lon, lat, idx) in tiendas_coords]
    if cerrada:
        vehicle = optimization.Vehicle(id=1, profile='driving-car', start=cd_coord, end=cd_coord)
    else:
        vehicle = optimization.Vehicle(id=1, profile='driving-car', start=cd_coord)
    result = client.optimization(jobs=jobs, vehicles=[vehicle], geometry=True)
    if not result.get('routes'):
        return None
    route = result['routes'][0]
    decoded = convert.decode_polyline(route['geometry'])
    coords_route = [(lat, lon) for lon, lat in decoded['coordinates']]
    steps = route.get('steps', [])
    orden_visita = [step['id'] for step in steps if step.get('type') == 'job']
    return {
        'orden': orden_visita,
        'coords_route': coords_route,
        'distance_km': route.get('distance', 0) / 1000.0,
        'duration_min': route.get('duration', 0) / 60.0
    }

try:
    kmeans_pretrained, knn_pretrained, scaler = cargar_modelos()
except Exception as e:
    st.error(f"Error al cargar los modelos: {type(e).__name__}: {e}")
    st.stop()

# ===========================================================
# SIDEBAR — UPLOAD
# ===========================================================
st.sidebar.header("📂 Cargar tu propio dataset")

with st.sidebar.expander("📥 Descarga la plantilla", expanded=False):
    st.caption("Edítala con tus tiendas y súbela abajo.")
    st.markdown("""
    | Columna | Ejemplo |
    |---|---|
    | codigo_sucursal | 101 |
    | name_sucursal | Tienda Centro |
    | distrito | Lima |
    | **latitud** ⭐ | -12.046374 |
    | **longitud** ⭐ | -77.042793 |
    """)
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.download_button("📊 Excel", data=generar_plantilla_excel(),
            file_name="plantilla_tiendas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
    with col_p2:
        st.download_button("📄 CSV", data=generar_plantilla_csv(),
            file_name="plantilla_tiendas.csv", mime="text/csv", use_container_width=True)

archivo_subido = st.sidebar.file_uploader("Sube un archivo Excel o CSV", type=["xlsx", "xls", "csv"])

if archivo_subido is not None:
    try:
        if archivo_subido.name.endswith(".csv"):
            df = pd.read_csv(archivo_subido)
        else:
            df = pd.read_excel(archivo_subido)
        if not {"latitud", "longitud"}.issubset(df.columns):
            st.sidebar.error("⚠️ Faltan columnas 'latitud' y 'longitud'.")
            st.stop()
        if "codigo_sucursal" not in df.columns:
            df["codigo_sucursal"] = range(1, len(df) + 1)
        if "name_sucursal" not in df.columns:
            df["name_sucursal"] = [f"Tienda {i}" for i in range(1, len(df) + 1)]
        if "distrito" not in df.columns:
            df["distrito"] = "Sin especificar"
        df = df.dropna(subset=["latitud", "longitud"]).reset_index(drop=True)
        if len(df) < 2:
            st.sidebar.error("⚠️ Al menos 2 tiendas válidas.")
            st.stop()
        st.sidebar.success(f"✅ {len(df)} tiendas cargadas")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        st.stop()
else:
    df = cargar_datos()
    st.sidebar.info(f"ℹ️ Dataset por defecto ({len(df)} tiendas)")

st.sidebar.markdown("---")

# ===========================================================
# HEADER
# ===========================================================
st.title("🏪 Agrupación de Tiendas y Ruteo Óptimo de Despacho")
st.markdown(f"""
**Alumno:** {NOMBRE_ALUMNO}
**Código ISIL:** {CODIGO_ISIL}
**Cuaderno de código (COLAB):** [Abrir en Google Colab]({URL_COLAB})
""")
st.markdown("---")

with st.expander("📋 Descripción del problema y modelo", expanded=False):
    st.markdown("""
    ### Problema de negocio
    Optimizar las **rutas de despacho** desde un Centro de Distribución (CD) hacia múltiples tiendas en Lima.
    Primero agrupamos las tiendas por cercanía geográfica con KMeans, y luego, para cada grupo,
    calculamos la **ruta óptima por calles reales** usando OpenRouteService.
    """)

# ===========================================================
# SIDEBAR — MODELO
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
    return max(sils, key=sils.get), sils

dataset_id = "subido" if archivo_subido is not None else "default"
k_optimo, sil_dict = calcular_k_optimo(X_scaled, dataset_id)
st.sidebar.info(f"💡 **K óptimo sugerido:** {k_optimo}\n\n(Silhouette = {sil_dict[k_optimo]:.3f})")

max_k_slider = min(MAX_K, len(df) - 1)
k_usuario = st.sidebar.slider("Selecciona el número de clusters (K):",
    min_value=2, max_value=max_k_slider, value=min(k_optimo, max_k_slider), step=1)
usar_optimo = st.sidebar.checkbox("Usar K óptimo automático", value=False)
K = k_optimo if usar_optimo else k_usuario

st.sidebar.markdown("---")

# ===========================================================
# SIDEBAR — CD Y RUTEO
# ===========================================================
st.sidebar.header("🚛 Centro de Distribución y Ruteo")
cd_lat = st.sidebar.number_input("Latitud del CD:", value=-12.046374, format="%.6f")
cd_lon = st.sidebar.number_input("Longitud del CD:", value=-77.042793, format="%.6f")
tipo_recorrido = st.sidebar.selectbox(
    "Tipo de recorrido:",
    options=["cerrado", "abierto"],
    format_func=lambda x: "🔁 Cerrado (CD → tiendas → CD)" if x == "cerrado" else "➡️ Abierto (CD → tiendas)"
)
api_key_ors = st.sidebar.text_input(
    "API Key de OpenRouteService:", type="password",
    help="Obtén tu key gratis en https://openrouteservice.org/dev/#/signup"
)
calcular_rutas = st.sidebar.button("🚛 Calcular rutas óptimas", use_container_width=True, type="primary")
st.sidebar.markdown("---")

# ===========================================================
# SIDEBAR — FILTROS
# ===========================================================
st.sidebar.header("🔍 Filtros de visualización")

modo_enfoque = st.sidebar.radio(
    "Modo de visualización:",
    options=["todos", "aislar", "solo_rutas"],
    format_func=lambda x: {
        "todos": "👁️ Mostrar todo",
        "aislar": "🎯 Aislar un cluster",
        "solo_rutas": "🛣️ Solo rutas (sin zonas)"
    }[x], index=0
)

if modo_enfoque == "aislar":
    cluster_aislado = st.sidebar.selectbox("Cluster a aislar:",
        options=list(range(K)), format_func=lambda x: f"Cluster {x}")
    clusters_visibles = [cluster_aislado]
else:
    clusters_visibles = st.sidebar.multiselect(
        "Clusters visibles:",
        options=list(range(K)), default=list(range(K)),
        format_func=lambda x: f"Cluster {x}"
    )

mostrar_zonas_check = st.sidebar.checkbox("Mostrar zonas de cobertura", value=True)
mostrar_rutas_check = st.sidebar.checkbox("Mostrar rutas calculadas", value=True)
mostrar_tiendas_check = st.sidebar.checkbox("Mostrar tiendas", value=True)
mostrar_numeros_orden = st.sidebar.checkbox("Mostrar números de orden", value=True)
mostrar_centroides = st.sidebar.checkbox("Mostrar centroides", value=True)
mostrar_zonas = mostrar_zonas_check and modo_enfoque != "solo_rutas"

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Estilo del mapa")
estilo_mapa = st.sidebar.selectbox(
    "Estilo:",
    options=["carto-positron", "open-street-map", "carto-darkmatter", "white-bg"],
    format_func=lambda x: {
        "carto-positron": "🌅 Claro (recomendado)",
        "open-street-map": "🗺️ OpenStreetMap",
        "carto-darkmatter": "🌃 Oscuro",
        "white-bg": "⬜ Blanco minimalista"
    }[x], index=0
)
paleta_colores = st.sidebar.selectbox("Paleta de colores:",
    options=["Vivid", "Bold", "Pastel", "Plotly", "D3", "Light24"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Ver detalles")
mostrar_codo = st.sidebar.checkbox("Mostrar método del codo / silhouette", value=True)
mostrar_metricas = st.sidebar.checkbox("Mostrar métricas del modelo", value=True)

# ===========================================================
# ENTRENAR MODELO
# ===========================================================
modelo = KMeans(n_clusters=K, random_state=42, n_init=10)
df["cluster"] = modelo.fit_predict(X_scaled)
df["cluster"] = df["cluster"].astype(str)

centroides = scaler.inverse_transform(modelo.cluster_centers_)
df_centroides = pd.DataFrame(centroides, columns=["latitud", "longitud"])
df_centroides["cluster"] = [f"Centroide {i}" for i in range(K)]

# ===========================================================
# CÁLCULO DE RUTAS
# ===========================================================
if "rutas_calculadas" not in st.session_state:
    st.session_state.rutas_calculadas = None
    st.session_state.rutas_k_config = None

if calcular_rutas:
    if not api_key_ors:
        st.sidebar.error("⚠️ Ingresa tu API Key de OpenRouteService.")
    else:
        config_actual = f"{K}_{cd_lat}_{cd_lon}_{tipo_recorrido}_{dataset_id}"
        with st.spinner("🚛 Calculando rutas óptimas por calles reales..."):
            rutas = {}
            errores = []
            for i in range(K):
                cluster_data = df[df["cluster"] == str(i)].copy()
                tiendas_coords = [(float(row["longitud"]), float(row["latitud"]), idx)
                                  for idx, row in cluster_data.iterrows()]
                try:
                    resultado = calcular_ruta_ors(
                        cd_coord=[cd_lon, cd_lat],
                        tiendas_coords=tiendas_coords,
                        api_key=api_key_ors,
                        cerrada=(tipo_recorrido == "cerrado")
                    )
                    if resultado:
                        rutas[i] = resultado
                except Exception as e:
                    errores.append(f"Cluster {i}: {e}")
            if errores:
                for err in errores:
                    st.sidebar.error(err)
            if rutas:
                st.session_state.rutas_calculadas = rutas
                st.session_state.rutas_k_config = config_actual
                st.sidebar.success(f"✅ {len(rutas)} rutas calculadas")

config_actual = f"{K}_{cd_lat}_{cd_lon}_{tipo_recorrido}_{dataset_id}"
rutas_validas = (
    st.session_state.rutas_calculadas is not None
    and st.session_state.rutas_k_config == config_actual
)

# ===========================================================
# MÉTRICAS
# ===========================================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Tiendas", len(df))
col2.metric("Clusters", K)
col3.metric("Silhouette", f"{silhouette_score(X_scaled, df['cluster']):.3f}")
col4.metric("Davies-Bouldin", f"{davies_bouldin_score(X_scaled, df['cluster']):.3f}")
if rutas_validas:
    total_km_visibles = sum(r['distance_km'] for i, r in st.session_state.rutas_calculadas.items()
                            if i in clusters_visibles)
    col5.metric("Km rutas visibles", f"{total_km_visibles:.1f} km")
else:
    col5.metric("Km totales rutas", "—")

st.markdown("---")

if not clusters_visibles:
    st.warning("⚠️ No hay clusters seleccionados. Activa al menos uno en el sidebar.")

# ===========================================================
# MAPA INTERACTIVO
# ===========================================================
st.subheader(f"🗺️ Mapa interactivo — {K} zonas de despacho")

if modo_enfoque == "aislar" and clusters_visibles:
    cluster_data_focus = df[df["cluster"] == str(clusters_visibles[0])]
    center_lat = cluster_data_focus["latitud"].mean()
    center_lon = cluster_data_focus["longitud"].mean()
    lat_range = cluster_data_focus["latitud"].max() - cluster_data_focus["latitud"].min()
    lon_range = cluster_data_focus["longitud"].max() - cluster_data_focus["longitud"].min()
else:
    center_lat = df["latitud"].mean()
    center_lon = df["longitud"].mean()
    lat_range = df["latitud"].max() - df["latitud"].min()
    lon_range = df["longitud"].max() - df["longitud"].min()

max_range = max(lat_range, lon_range, 0.001)
if max_range < 0.02: zoom_calc = 14
elif max_range < 0.05: zoom_calc = 13
elif max_range < 0.1: zoom_calc = 12
elif max_range < 0.3: zoom_calc = 11
elif max_range < 1: zoom_calc = 9
elif max_range < 5: zoom_calc = 6
elif max_range < 20: zoom_calc = 4
else: zoom_calc = 2

paletas = {
    "Vivid": px.colors.qualitative.Vivid, "Bold": px.colors.qualitative.Bold,
    "Pastel": px.colors.qualitative.Pastel, "Plotly": px.colors.qualitative.Plotly,
    "D3": px.colors.qualitative.D3, "Light24": px.colors.qualitative.Light24
}
colores = paletas[paleta_colores]

fig_mapa = go.Figure()

# 1) Zonas de cobertura
if mostrar_zonas:
    for i in range(K):
        if i not in clusters_visibles:
            continue
        cluster_data = df[df["cluster"] == str(i)]
        if len(cluster_data) >= 3:
            puntos = cluster_data[["longitud", "latitud"]].values
            try:
                hull = ConvexHull(puntos)
                hull_pts = puntos[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                color_cluster = colores[i % len(colores)]
                fig_mapa.add_trace(go.Scattermapbox(
                    lat=hull_pts[:, 1], lon=hull_pts[:, 0],
                    mode="lines", fill="toself",
                    fillcolor=hex_to_rgba(color_cluster, alpha=0.15),
                    line=dict(color=color_cluster, width=2),
                    name=f"Zona {i}", hoverinfo="skip", showlegend=False
                ))
            except Exception:
                pass

# 2) Rutas óptimas
if rutas_validas and mostrar_rutas_check:
    for i, ruta in st.session_state.rutas_calculadas.items():
        if i not in clusters_visibles:
            continue
        color_cluster = colores[i % len(colores)]
        lats = [c[0] for c in ruta['coords_route']]
        lons = [c[1] for c in ruta['coords_route']]
        fig_mapa.add_trace(go.Scattermapbox(
            lat=lats, lon=lons, mode="lines",
            line=dict(color=color_cluster, width=4),
            name=f"Ruta {i}", hoverinfo="skip", showlegend=False
        ))

# 3) TIENDAS — diseño badge tipo "Material Design"
# Capa exterior: marker grande color del cluster
# Capa interior: marker pequeño blanco con número del orden adentro
if mostrar_tiendas_check:
    for i in range(K):
        if i not in clusters_visibles:
            continue
        cluster_data = df[df["cluster"] == str(i)].copy()
        color_cluster = colores[i % len(colores)]

        # Determinar si hay orden numerado para este cluster
        tiene_orden = (rutas_validas and mostrar_numeros_orden
                       and i in st.session_state.rutas_calculadas)

        if tiene_orden:
            orden = st.session_state.rutas_calculadas[i]['orden']
            mapa_orden = {idx_global: pos + 1 for pos, idx_global in enumerate(orden)}
            cluster_data["orden_visita"] = cluster_data.index.map(mapa_orden).fillna(0).astype(int)
            cluster_data = cluster_data.sort_values("orden_visita")
        else:
            cluster_data["orden_visita"] = 0

        # Capa exterior: anillo del cluster (más grande)
        fig_mapa.add_trace(go.Scattermapbox(
            lat=cluster_data["latitud"], lon=cluster_data["longitud"],
            mode="markers",
            marker=dict(
                size=18 if tiene_orden else 13,
                color=color_cluster,
                opacity=1.0
            ),
            name=f"Cluster {i}",
            customdata=cluster_data[["name_sucursal", "codigo_sucursal", "distrito", "orden_visita"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Código: %{customdata[1]}<br>"
                "Distrito: %{customdata[2]}<br>"
                "Orden de visita: %{customdata[3]}<br>"
                "Lat: %{lat:.5f}<br>"
                "Lon: %{lon:.5f}<extra></extra>"
            )
        ))

        # Capa interior: badge blanco con número (solo cuando hay rutas)
        if tiene_orden:
            fig_mapa.add_trace(go.Scattermapbox(
                lat=cluster_data["latitud"], lon=cluster_data["longitud"],
                mode="markers+text",
                marker=dict(size=11, color="white", opacity=1.0),
                text=cluster_data["orden_visita"].astype(str),
                textfont=dict(size=9, color="#000000", family="Arial Black"),
                textposition="middle center",
                showlegend=False, hoverinfo="skip"
            ))

# 4) Centroides
if mostrar_centroides:
    indices_visibles = [i for i in range(K) if i in clusters_visibles]
    centroides_visibles = df_centroides.iloc[indices_visibles] if indices_visibles else df_centroides.iloc[0:0]
    if len(centroides_visibles) > 0:
        fig_mapa.add_trace(go.Scattermapbox(
            lat=centroides_visibles["latitud"], lon=centroides_visibles["longitud"],
            mode="markers", marker=dict(size=20, color="#1a1a1a"),
            hoverinfo="skip", showlegend=False
        ))
        fig_mapa.add_trace(go.Scattermapbox(
            lat=centroides_visibles["latitud"], lon=centroides_visibles["longitud"],
            mode="markers+text",
            marker=dict(size=16, color="white"),
            text=[f"C{i}" for i in indices_visibles],
            textfont=dict(size=9, color="#000000", family="Arial Black"),
            textposition="middle center",
            name="📍 Centroide",
            hovertext=[f"<b>Centroide Cluster {i}</b>" for i in indices_visibles],
            hoverinfo="text"
        ))

# 5) Centro de Distribución
fig_mapa.add_trace(go.Scattermapbox(
    lat=[cd_lat], lon=[cd_lon],
    mode="markers", marker=dict(size=28, color="#000000"),
    hoverinfo="skip", showlegend=False
))
fig_mapa.add_trace(go.Scattermapbox(
    lat=[cd_lat], lon=[cd_lon],
    mode="markers+text",
    marker=dict(size=22, color="#FFD700"),
    text=["🏭"], textfont=dict(size=14),
    textposition="middle center",
    name="🏭 Centro de Distribución",
    hovertext=[f"<b>Centro de Distribución</b><br>Lat: {cd_lat:.5f}<br>Lon: {cd_lon:.5f}"],
    hoverinfo="text"
))

fig_mapa.update_layout(
    mapbox=dict(style=estilo_mapa, center=dict(lat=center_lat, lon=center_lon), zoom=zoom_calc),
    height=680, margin={"r": 0, "t": 10, "l": 0, "b": 0},
    legend=dict(
        title=dict(text="<b>Leyenda</b>", font=dict(size=13, color="#000000")),
        yanchor="top", y=0.99, xanchor="left", x=0.01,
        bgcolor="rgba(255,255,255,0.95)", bordercolor="#1a1a1a", borderwidth=1,
        font=dict(size=12, color="#000000", family="Arial"),
        itemsizing="constant"
    ),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    # HOVER con texto bien legible y nítido
    hoverlabel=dict(
        bgcolor="rgba(255,255,255,0.98)",
        bordercolor="#1a1a1a",
        font=dict(size=14, color="#000000", family="Arial")
    )
)

st.plotly_chart(fig_mapa, use_container_width=True)

if modo_enfoque == "aislar":
    st.caption(f"🎯 **Modo aislado:** Cluster {clusters_visibles[0]}. Cambia el modo en el sidebar.")
elif modo_enfoque == "solo_rutas":
    st.caption("🛣️ **Modo solo rutas:** zonas de cobertura ocultas.")
elif rutas_validas:
    st.caption("💡 🏭 CD (dorado) = punto de partida. **Líneas gruesas** = ruta por calles. **Círculos blancos numerados** = orden de visita.")
else:
    st.caption("💡 Calcula las rutas óptimas desde el sidebar para ver el ruteo.")

# ===========================================================
# DETALLE DE RUTAS
# ===========================================================
if rutas_validas:
    st.markdown("---")
    st.subheader("🚛 Detalle de rutas óptimas")
    resumen_rutas = []
    for i, ruta in st.session_state.rutas_calculadas.items():
        cluster_data = df[df["cluster"] == str(i)]
        resumen_rutas.append({
            "Cluster": i,
            "Visible": "✅" if i in clusters_visibles else "—",
            "Tiendas": len(cluster_data),
            "Distancia (km)": round(ruta['distance_km'], 2),
            "Duración (min)": round(ruta['duration_min'], 1)
        })
    df_resumen_rutas = pd.DataFrame(resumen_rutas)
    df_resumen_rutas.loc[len(df_resumen_rutas)] = [
        "TOTAL", "", df_resumen_rutas["Tiendas"].sum(),
        round(df_resumen_rutas["Distancia (km)"].sum(), 2),
        round(df_resumen_rutas["Duración (min)"].sum(), 1)
    ]
    st.dataframe(df_resumen_rutas, use_container_width=True, hide_index=True)

    rutas_visibles_dict = {i: r for i, r in st.session_state.rutas_calculadas.items()
                           if i in clusters_visibles}
    if rutas_visibles_dict:
        tabs = st.tabs([f"Cluster {i}" for i in rutas_visibles_dict.keys()])
        for tab_idx, (i, ruta) in enumerate(rutas_visibles_dict.items()):
            with tabs[tab_idx]:
                cluster_data = df[df["cluster"] == str(i)].copy()
                orden = ruta['orden']
                secuencia = []
                for pos, idx_global in enumerate(orden, start=1):
                    if idx_global in cluster_data.index:
                        row = cluster_data.loc[idx_global]
                        secuencia.append({
                            "Orden": pos,
                            "Código": row["codigo_sucursal"],
                            "Tienda": row["name_sucursal"],
                            "Distrito": row["distrito"],
                            "Latitud": round(row["latitud"], 5),
                            "Longitud": round(row["longitud"], 5)
                        })
                df_seq = pd.DataFrame(secuencia)
                st.markdown(f"**Distancia:** {ruta['distance_km']:.2f} km — **Duración:** {ruta['duration_min']:.1f} min")
                st.dataframe(df_seq, use_container_width=True, hide_index=True)
                csv_ruta = df_seq.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"⬇️ Descargar ruta Cluster {i} (CSV)",
                    data=csv_ruta, file_name=f"ruta_cluster_{i}.csv",
                    mime="text/csv", key=f"download_ruta_{i}"
                )

# ===========================================================
# RESTO
# ===========================================================
st.markdown("---")
st.subheader("📊 Resumen general por cluster")
resumen = df.groupby("cluster").agg(
    cantidad_tiendas=("codigo_sucursal", "count"),
    lat_centro=("latitud", "mean"),
    lon_centro=("longitud", "mean")
).reset_index()
st.dataframe(resumen, use_container_width=True)

if mostrar_codo:
    st.subheader("📈 Selección del K óptimo")
    inercias, silhouettes = [], []
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
                           labels={"x": "K", "y": "Inercia (WCSS)"}, title="Método del Codo")
        fig_codo.add_vline(x=K, line_dash="dash", line_color="red", annotation_text=f"K = {K}")
        st.plotly_chart(fig_codo, use_container_width=True)
    with col_b:
        fig_sil = px.line(x=K_range, y=silhouettes, markers=True,
                          labels={"x": "K", "y": "Silhouette Score"}, title="Silhouette por K")
        fig_sil.add_vline(x=K, line_dash="dash", line_color="red", annotation_text=f"K = {K}")
        st.plotly_chart(fig_sil, use_container_width=True)

if mostrar_metricas:
    st.subheader("📐 Interpretación de las métricas")
    st.markdown("""
    | Métrica | Valor | Interpretación |
    |---|---|---|
    | **Silhouette Score** | {:.3f} | Va de -1 a 1. Más cercano a 1 = clusters mejor separados. |
    | **Davies-Bouldin** | {:.3f} | Más bajo = mejor (clusters compactos y separados). |
    | **Inercia (WCSS)** | {:.4f} | Suma de distancias al centroide. |
    """.format(
        silhouette_score(X_scaled, df["cluster"]),
        davies_bouldin_score(X_scaled, df["cluster"]),
        modelo.inertia_
    ))

st.markdown("---")
st.subheader("📋 Detalle de tiendas asignadas a cada cluster")
df_descarga = df[["codigo_sucursal", "name_sucursal", "distrito",
                  "latitud", "longitud", "cluster"]].copy().sort_values(["cluster", "name_sucursal"])
st.dataframe(df_descarga, use_container_width=True, height=400)
csv = df_descarga.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar resultados (CSV)", data=csv,
    file_name=f"tiendas_agrupadas_K{K}.csv", mime="text/csv")

st.markdown("---")
st.subheader("🆕 Predecir el cluster de una tienda NUEVA (modelo KNN)")
st.caption("Ingresa las coordenadas y el clasificador KNN te dirá a qué grupo pertenece.")
col_n1, col_n2, col_n3 = st.columns([1, 1, 1])
with col_n1:
    nueva_lat = st.number_input("Latitud", value=-12.18, format="%.6f", key="pred_lat")
with col_n2:
    nueva_lon = st.number_input("Longitud", value=-76.96, format="%.6f", key="pred_lon")
with col_n3:
    if st.button("🔍 Predecir cluster", use_container_width=True):
        from sklearn.neighbors import KNeighborsClassifier
        knn_actual = KNeighborsClassifier(n_neighbors=3)
        knn_actual.fit(X_scaled, df["cluster"].astype(int))
        nueva_coord = scaler.transform([[nueva_lat, nueva_lon]])
        pred = knn_actual.predict(nueva_coord)[0]
        st.success(f"La nueva tienda pertenece al **Cluster {pred}**")

st.markdown("---")
st.caption(f"""
Proyecto académico — Proceso de Aprendizaje 2 — ISIL
Modelos: KMeans + KNN + Ruteo con OpenRouteService
[Ver cuaderno de código (Google Colab)]({URL_COLAB})
""")
