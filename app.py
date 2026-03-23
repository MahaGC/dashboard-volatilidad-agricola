import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. CONFIGURACIÓN Y ESTILOS
# ==========================================
st.set_page_config(
    page_title="Dashboard Volatilidad Agrícola - Equipo 05C",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #0E1117 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

COLORS = {
    "Maize": "#FBC02D",      # Amarillo Maíz
    "Coffee": "#EB4828",     # Café
    "Cocoa": "#4E342E",      # Marrón Cacao
    "Sugar": "#29B6F6",      # Azul Commodity
    "Soybeans": "#43A047"    # Verde Soya
}

ICONOS = {"Maize": "🌽", "Coffee": "☕", "Cocoa": "🍫", "Sugar": "🍬", "Soybeans": "🌱"}

# Inyectar CSS para tablas y métricas
st.markdown("""
    <style>
    /* Fondo general de la app */
    .stApp {
        background-color: #0E1117;
    }
            
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0E1117 !important;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }
            
    /* Inputs */
    input, textarea {
        background-color: #1E293B !important;
        color: white !important;
    }

    /* Contenedor de la lista de pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    /* Estilo de cada pestaña individual (No seleccionada) */
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1E293B; /* Azul grisáceo oscuro */
        border-radius: 8px 8px 0px 0px;
        padding: 10px 25px;
        color: #FFFFFF !important; /* Texto blanco forzado */
        font-weight: 500;
        border: 1px solid #334155;
    }

    /* Estilo de la pestaña cuando pasas el mouse (Hover) */
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(180deg, rgba(67, 160, 71, 0.2) 0%, rgba(67, 160, 71, 0.05) 100%) !important;
        color: #43A047 !important; /* Cambia a amarillo al pasar el mouse */
        border: 1px solid rgba(67, 160, 71, 0.3) 
    }

    /* Estilo de la pestaña SELECCIONADA */
    .stTabs [aria-selected="true"] {
        background-color: #43A047 !important; /* Azul brillante para resaltar */
        color: #FFFFFF !important;
        font-weight: bold;
        border-bottom: 3px solid #FFFFFF !important;
    }
    
    /* Ajuste para las métricas para que se vean en fondo oscuro */
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
    }
            
    /*Contenedor principal del KPI*/
    .kpi-card{
        background-color: #1E293B;
        border-radius: 15px;
        padding: 1.5rem 1 rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.4 cubic-bezier(0.25, 0.8, 0.25, 1);
        text-align: center;
        margin-bottom: 10px;
            
        /* RESPONSIVIDAD CLAVE */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 180px;
        width: 100%;
    }
            
    .kpi-card:hover{
        transform: translateY(-5px);
        background: linear-gradient(135deg, rgba(67, 210, 71, 0.36) 0%, rgba(67, 160, 71, 0.05) 100%) !important;
        border: 1px solid rgba(67, 160, 71, 0.4);
        box-shadow: 0 12px 20px rgba(0,0,0,0.4);
    }
            
    .kpi-icon{
        font-size: 35px;
        margin-bottom: 10px;
        display: block;
    }
            
    .kpi-label {
        color: #B0B0B0;
        font-size: 14px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
            
    .kpi-value {
        color: #FFFFFF;
        font-size: 24px;
        font-weight: bold;
       
             margin: 5px 0;
    }
    .kpi-delta {
        font-size: 14px;
        font-weight: bold;
    }

    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. FUNCIÓN DE LIMPIEZA DEL DATASET
# ==========================================
@st.cache_data
def limpiar_dataset(ruta_archivo):
    df = pd.read_excel(
        ruta_archivo,
        sheet_name="Annual Prices (Nominal)",
        skiprows=6
    )

    # Limpiar nombres de columnas
    df.columns = df.columns.astype(str).str.strip()

    # Renombrar primera columna
    df = df.rename(columns={df.columns[0]: "año"})

    df["año"] = pd.to_numeric(df["año"], errors="coerce")
    df = df.dropna(subset=["año"])

    columnas = df.columns.str.lower()

    # Detectar columnas automáticamente
    maize_col = df.columns[columnas.str.contains("maize")]
    coffee_col = df.columns[columnas.str.contains("coffee")]
    cocoa_col = df.columns[columnas.str.contains("cocoa")]
    sugar_col = df.columns[columnas.str.contains("sugar")]
    soy_col = df.columns[columnas.str.contains("soy")]

    df_cultivos = df[
        [
            "año",
            maize_col[0],
            coffee_col[0],
            cocoa_col[0],
            sugar_col[0],
            soy_col[0],
        ]
    ].copy()

    df_cultivos.columns = [
        "año",
        "Maize",
        "Coffee",
        "Cocoa",
        "Sugar",
        "Soybeans",
    ]

    for col in df_cultivos.columns[1:]:
        df_cultivos[col] = pd.to_numeric(df_cultivos[col], errors="coerce")

    return df_cultivos

# --- CARGAR DATASET ---
try:
    df_original = limpiar_dataset("data/pink_sheet.xlsx")
except Exception as e:
    st.error(f"Error al cargar el archivo: {e}. Asegúrate de que 'data/pink_sheet.xlsx' existe.")
    st.stop()

# ==========================================
# 3. SIDEBAR (FILTROS GLOBALES)
# ==========================================
with st.sidebar:
    st.title("🛠️ Configuración Global")
    st.markdown("---")
    
    cultivos_sel = st.multiselect(
        "Seleccione cultivos para analizar:",
        options=list(COLORS.keys()),
        default=["Maize", "Coffee", "Cocoa", "Sugar", "Soybeans"]
    )
    
    year_range = st.slider(
        "Periodo de tiempo:",
        min_value=int(df_original["año"].min()),
        max_value=int(df_original["año"].max()),
        value=(int(df_original["año"].min()), int(df_original["año"].max()))
    )
    
    st.info("📌 Esta selección afecta a todas las pestañas.")

# Filtrado de datos global
df_view = df_original[(df_original["año"] >= year_range[0]) & (df_original["año"] <= year_range[1])]

if not cultivos_sel:
    st.warning("⚠️ Por favor, seleccione al menos un cultivo en la barra lateral izquierda.")
    st.stop()

# ==========================================
# 4. DISTRIBUCIÓN PRINCIPAL - TABS
# ==========================================
st.title("🌾 Análisis Predictivo de Precios Agrícolas")

tab1, tab2, tab3 = st.tabs(["📊 Comparativa Global", "🔍 Detalle de Mercado", "🔮 Modelado Predictivo"])

# --- TAB 1: COMPARATIVA ---
with tab1:
    st.subheader("Indicadores de Volatilidad Histórica")
    
    # Métricas dinámicas
    cols = st.columns(len(cultivos_sel))
    for i, crop in enumerate(cultivos_sel):
        volatilidad = df_view[crop].std()
        promedio = df_view[crop].mean()
        cv = (volatilidad / promedio) * 100 

        #Iconos y color por cultivo
        icono = ICONOS.get(crop,"🌾")
        color_cultivo = COLORS.get(crop,"#43A047")

        with cols[i]:
            st.markdown(f"""
                <div class="kpi-card">
                    <span class="kpi-icon">{icono}</span>
                    <div class="kpi-label">Volatilidad {crop}</div>
                    <div class="kpi-value">{volatilidad:.2f} USD</div>
                    <div class="kpi-delta" style="color: #FBC02D;">
                        {cv:.1f}% Coef. Variación
                    </div>
                </div>
            """, unsafe_allow_html=True)
                
    st.markdown("---")
    
    # Gráfica de líneas (Multivariada)
    fig_comp = px.line(
        df_view, x="año", y=cultivos_sel,
        color_discrete_map=COLORS,
        labels={"value": "Precio (USD)", "variable": "Cultivo"},
        title="Evolución Histórica Comparativa",
        markers=True
    )

    fig_comp.update_layout(
        hovermode="x unified", 
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=60, b=0),

        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),

        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=13,
            font_family="Inter, sans-serif",
            font_color="white",
            bordercolor="#334155"
        )
    )

    fig_comp.update_traces(
        hovertemplate="<b>%{fullData.name}</b>: %{y:.2f} USD<extra></extra>"
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)

# --- TAB 2: DETALLE INDIVIDUAL ---
with tab2:
    st.subheader("Análisis de Tendencias y Distribución")
    crop_focus = st.selectbox(
        "Seleccione un cultivo para inspección profunda:", 
        cultivos_sel,
        format_func=lambda x: f"{ICONOS.get(x, '')} {x}"
    )
    
    col_a, col_b = st.columns([2, 1])

    with col_a:
        # Gráfica de área para ver volumen/tendencia
        fig_area = px.area(
            df_view, 
            x="año", 
            y=crop_focus, 
            title=f"Tendencia Histórica: {crop_focus}", 
            template="plotly_dark",
            markers=True
        )
        
        fig_area.update_layout(
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white", family="Inter, sans-serif"),
            margin=dict(l=0, r=20, t=50, b=0),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#334155", zeroline=False),
            hoverlabel=dict(
                bgcolor="#1E293B", 
                font_size=13,
                font_color="white", 
                bordercolor=COLORS[crop_focus]
            )
        )

        fig_area.update_traces(
            line_color=COLORS[crop_focus], 
            line_width=3,                 
            fill='tozeroy',              
            fillcolor='rgba(128, 128, 128, 0.1)', 
            marker=dict(
                size=7, 
                color="#FFFFFF",        
                line=dict(color=COLORS[crop_focus], width=1.5)
            ),
            hovertemplate="<b>Año:</b> %{x}<br><b>Precio:</b> %{y:.2f} USD<extra></extra>"
        )

        st.plotly_chart(fig_area, use_container_width=True)
        
    with col_b:
        st.write(f"**Estadísticas de {ICONOS[crop_focus]} {crop_focus}**")

        stats_df = df_view[crop_focus].describe().to_frame().T
        
        st.dataframe(
            df_view[crop_focus].describe(), 
            use_container_width=True,
            height=300
        )

        st.info(f"El valor máximo alcanzado por el {crop_focus} fue de **{df_view[crop_focus].max():.2f} USD**.")

# --- TAB 3: PREDICCIÓN ---
with tab3: 
    st.subheader("Predicción de Precios Agrícolas")

    # 1. Selector de cultivo para predicción
    cultivo_pred = st.selectbox(
        "Selecciona cultivo para predicción",
        cultivos_sel,
        format_func=lambda x: f"{ICONOS.get(x, '')} {x}"
    )

    # --- CÁLCULOS DEL MODELO ---
    df_model = df_view[["año", cultivo_pred]].dropna()
    X = df_model[["año"]]
    y = df_model[[cultivo_pred]]
    modelo = LinearRegression().fit(X, y)
    rmse = np.sqrt(mean_squared_error(y, modelo.predict(X)))

    # Cálculos base
    precio_prom = df_view[cultivo_pred].mean()
    precio_max = df_view[cultivo_pred].max()
    precio_min = df_view[cultivo_pred].min()
    precio_actual = df_view[cultivo_pred].iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)

    # Diccionario temporal para iterar las tarjetas
    datos_kpi = [
        {"label": "Precio Promedio", "val": precio_prom, "icon": "💰"},
        {"label": "Precio Máximo", "val": precio_max, "icon": "📈"},
        {"label": "Precio Mínimo", "val": precio_min, "icon": "📉"},
        {"label": "Último Precio", "val": precio_actual, "icon": "📅"},
        {"label": "ERROR RMSE", "val": rmse, "icon": "🎯", "color": "#D32F2F"}
    ]

    for i, kpi in enumerate(datos_kpi):
        with [col1, col2, col3, col4, col5][i]:
            st.markdown(f"""
                <div class="kpi-card" padding: 15px;">
                    <span class="kpi-icon">{kpi['icon']}</span>
                    <div class="kpi-label" style="font-size: 0.7rem; opacity: 0.8;">{kpi['label']}</div>
                    <div class="kpi-value" style="font-size: 1.2rem;">{kpi['val']:.2f} USD</div>
                </div>
            """, unsafe_allow_html=True)

    # --- NUEVA FILA: RMSE + NOTA METODOLÓGICA ---
    st.markdown("<br>", unsafe_allow_html=True) # Espaciado sutil
    with st.expander("📝 Nota Metodológica", expanded=False):
        st.markdown(f"""
        El modelo ha sido entrenado con datos históricos de **{df_model['año'].min()} a {df_model['año'].max()}**. 
        La proyección utiliza una regresión lineal para estimar la tendencia a 5 años. 
        El RMSE ({rmse:.2f}) representa la desviación promedio de los datos reales respecto a la línea de tendencia.
        """)

    # 4. PREDICCIÓN FUTURA
    años_futuros = np.arange(df_view["año"].max() + 1, df_view["año"].max() + 6)
    df_futuro = pd.DataFrame({"año": años_futuros})
    predicciones = modelo.predict(df_futuro)
    df_pred = pd.DataFrame({"año": años_futuros, "Predicción": predicciones.flatten()})

    # 5. GRÁFICA DE PREDICCIÓN
    fig_pred = px.line(
        df_view, x="año", y=cultivo_pred,
        title=f"Predicción del precio de {cultivo_pred}",
        markers=True,
        template="plotly_dark"
    )

    fig_pred.add_scatter(
        x=df_pred["año"], 
        y=df_pred["Predicción"],
        mode="lines+markers", 
        name="Predicción Futura",
        line=dict(color="#D32F2F", width=3, dash="dot"),
        marker=dict(symbol="diamond", size=8, color="#D32F2F"),
        hovertemplate="<b>Predicción:</b> %{y:.2f} USD<extra></extra>"
    )

    # Estilización del Tooltip y Fondo
    fig_pred.update_layout(
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_color="white",
            bordercolor="#334155"
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, title="Año"),
        yaxis=dict(showgrid=True, gridcolor="#334155", zeroline=False, title="Precio (USD)"),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom", 
            y=1.02, 
            xanchor="right",
            x=1
        )
    )

    fig_pred.update_traces(
        line=dict(color=COLORS.get(cultivo_pred, "#43A047"), width=3),
        fill='tozeroy',
        fillcolor='rgba(128, 128, 128, 0.05)', 
        marker=dict(size=6, color="white", line=dict(width=1, color=COLORS.get(cultivo_pred))),
        name="Histórico",
        hovertemplate="<b>Precio Real:</b> %{y:.2f} USD<extra></extra>"
    )

    # Línea vertical de inicio de predicción
    fig_pred.add_vline(
        x=df_view["año"].max(), 
        line_dash="dash", 
        line_color="gray",
        annotation_text="Inicio Predicción",
        annotation_position="top right"
    )

    st.plotly_chart(fig_pred, use_container_width=True)

st.markdown("---")
st.caption("Prototipo - Equipo 05C | Maestría en Análisis y Visualización de Datos Masivos")