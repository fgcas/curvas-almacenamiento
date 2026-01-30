import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Curvas Hidráulicas", layout="wide")
st.title("🌊 Generador de Ecuaciones: Cota - Volumen")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stCode { font-family: 'Courier New', monospace; }
    div.block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# --- CONFIGURACIÓN DE RUTA ---
FOLDER_PATH = "data"
FILE_NAME = "data_cota_volumen.csv"
FILE_PATH = os.path.join(FOLDER_PATH, FILE_NAME)

# --- VARIABLES GLOBALES ---
df_filtrado = None
cota_min_ref = 0.0
cota_max_ref = 0.0

# --- 1. CARGA DE DATOS AUTOMÁTICA ---
st.sidebar.header("1. Panel de Control")

if os.path.exists(FILE_PATH):
    try:
        # Cargar CSV desde la carpeta
        df_all = pd.read_csv(FILE_PATH)
        
        # Validar columnas
        req_cols = ['ID_Poza', 'Cota', 'Volumen']
        if not all(col in df_all.columns for col in req_cols):
            st.error(f"⚠️ El CSV debe tener las columnas: {req_cols}")
            st.stop()

        # --- SELECTOR DE POZA ---
        lista_pozas = df_all['ID_Poza'].unique()
        poza_seleccionada = st.sidebar.selectbox("Seleccionar Poza:", lista_pozas)
        
        # Filtrar datos
        df_filtrado = df_all[df_all['ID_Poza'] == poza_seleccionada].copy()
        df_filtrado = df_filtrado.sort_values(by='Cota')
        
        # --- CÁLCULO DE LÍMITES ---
        cota_min_ref = df_filtrado['Cota'].min()
        cota_max_ref = df_filtrado['Cota'].max()
        
        # Variable desplazada (X - Xmin)
        df_filtrado['X_Shift'] = df_filtrado['Cota'] - cota_min_ref

        # Mostrar tabla resumen en Sidebar
        st.sidebar.success(f"✅ Datos cargados: {len(df_filtrado)} registros")
        st.sidebar.markdown(f"**Rango Cota:** {cota_min_ref} - {cota_max_ref} m")
        st.sidebar.dataframe(df_filtrado[['Cota', 'Volumen']], height=500, use_container_width=True)

    except Exception as e:
        st.error(f"Error leyendo el archivo: {e}")
        st.stop()
else:
    # Instrucciones si no existe el archivo
    st.warning(f"⚠️ No se encuentra el archivo de datos.")
    st.info(f"""
    **Instrucciones de configuración:**
    1. Crea una carpeta llamada `{FOLDER_PATH}` en el mismo lugar que este script.
    2. Guarda tu archivo CSV con el nombre `{FILE_NAME}` dentro de esa carpeta.
    3. Asegúrate de que el CSV tenga las columnas: `ID_Poza`, `Cota`, `Volumen`.
    """)
    st.stop()

# --- 2. LÓGICA PRINCIPAL (Si hay datos) ---
if df_filtrado is not None:
    
    # Preparar variables
    x_real = df_filtrado['Cota']
    x_shift = df_filtrado['X_Shift']
    y_real = df_filtrado['Volumen']

    # --- A. GRÁFICO (ANCHO COMPLETO) ---
    st.subheader(f"📊 Curva Característica: {poza_seleccionada}")
    
    fig = go.Figure()
    
    # Puntos Reales
    fig.add_trace(go.Scatter(x=x_real, y=y_real, mode='markers', name='Datos Reales',
                             marker=dict(size=10, color='#00CC96', line=dict(width=1, color='black'))))
    
    # Rango suave
    x_range = np.linspace(cota_min_ref, cota_max_ref, 200)
    x_range_shift = x_range - cota_min_ref 
    
    resultados = [] # Lista para guardar ecuaciones

    # Calcular Polinomios (Grado 1 a 6)
    for grado in range(1, 7):
        try:
            coefs = np.polyfit(x_shift, y_real, grado)
            model = np.poly1d(coefs)
            
            # R2
            y_pred = model(x_shift)
            r2 = r2_score(y_real, y_pred)
            
            # Línea curva
            y_curve = model(x_range_shift)
            fig.add_trace(go.Scatter(
                x=x_range, y=y_curve, mode='lines', 
                name=f'Grado {grado} (R²={r2:.5f})',
                # visible='legendonly' if grado > 3 else True
            ))
            
            # Crear String de Ecuación
            terms = []
            for i, c in enumerate(coefs):
                potencia = grado - i
                signo = "+" if c >= 0 else "-"
                val_abs = abs(c)
                
                # Formateo condicional para limpieza visual
                if i == 0: # Primer término sin signo extra si es positivo
                    term_str = f"({c:.12f} * (x - {cota_min_ref})^{potencia})"
                else:
                    if potencia > 0:
                        term_str = f"{signo} ({val_abs:.12f} * (x - {cota_min_ref})^{potencia})"
                    else:
                        term_str = f"{signo} ({val_abs:.12f})" # Término independiente
                
                terms.append(term_str)
            
            ec_str = " ".join(terms)
            # Limpieza del primer signo si quedó raro
            if ec_str.startswith("+ "): ec_str = ec_str[2:]
            
            resultados.append((grado, r2, ec_str))
            
        except:
            pass

    fig.update_layout(height=500, template="plotly_white", 
                      xaxis_title="Cota (m.s.n.m)", yaxis_title="Volumen (m³)",
                      hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- B. ECUACIONES (DEBAJO DEL GRÁFICO) ---
    st.subheader("📐 Ecuaciones Generadas")
    st.caption(f"Nota: En estas fórmulas, **x** representa la Cota. El ajuste se realizó restando la cota mínima detectada ({cota_min_ref}).")

    for g, r2, ec in resultados:
        with st.expander(f"Polinomio Grado {g} | R² = {r2:.10f}", expanded=True):
            st.code(f"y = {ec}", language="text")

    st.divider()

    # --- C. INTERPOLADORA LINEAL ---
    st.subheader("🧮 Calculadora Rápida (Interpolación)")
    
    col_cal1, col_cal2 = st.columns([1, 2])
    
    with col_cal1:
        cota_input = st.number_input(
            "Ingresar Cota:", 
            min_value=float(cota_min_ref) - 10, 
            max_value=float(cota_max_ref) + 10,
            value=float(cota_min_ref),
            format="%.3f"
        )
    
    with col_cal2:
        vol_interp = np.interp(cota_input, df_filtrado['Cota'], df_filtrado['Volumen'])
        st.metric("Volumen Interpolado", f"{vol_interp:,.3f} m³")