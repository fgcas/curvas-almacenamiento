import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from sklearn.metrics import r2_score

# Configuración de página
st.set_page_config(page_title="Curvas Hidráulicas de almacenamiento", layout="wide")
st.title("🌊 Generador de Ecuaciones de llenado de reservorios")

# --- CSS para ajustar el ancho de la visualización del código ---
st.markdown("""
<style>
    .stCode { font-family: 'Courier New', monospace; }
</style>
""", unsafe_allow_html=True)

# --- SECCIÓN 1: INGRESO DE DATOS ---
st.sidebar.header("1. Datos de Entrada")
texto_input = st.sidebar.text_area("Pegar datos (Cota | Volumen) sin encabezados:", height=500)

df = None

if texto_input:
    try:
        # Procesamos el texto tabulado
        df = pd.read_csv(io.StringIO(texto_input), sep='\t', header=None)
        df.columns = ['Cota', 'Volumen']
        df = df.sort_values(by='Cota')
        
        # --- EL TRUCO: Calcular la Cota Mínima ---
        cota_min = df['Cota'].min()
        # Creamos la variable desplazada para el ajuste
        df['X_Shift'] = df['Cota'] - cota_min
        
        st.sidebar.success(f"✅ Datos cargados. Cota Mínima detectada: {cota_min}")
    except Exception as e:
        st.sidebar.error(f"Error leyendo datos: {e}")

# --- SECCIÓN 2: CÁLCULO Y VISUALIZACIÓN ---
if st.button("📊 Calcular Modelos (Grados 1-6)"):
    if df is not None:
        x_real = df['Cota']
        x_shift = df['X_Shift'] # Usamos esta para el fit
        y_real = df['Volumen']
        cota_min = df['Cota'].min()

        # Contenedor para el gráfico
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_real, y=y_real, mode='markers', name='Datos Reales',
                                 marker=dict(size=10, color='lime')))
        
        # Rango para graficar curvas suaves
        x_range = np.linspace(x_real.min(), x_real.max(), 200)
        x_range_shift = x_range - cota_min # Rango desplazado para predecir
        
        st.header("Resultados de Ajuste")
        st.markdown(f"**Nota:** En todas las ecuaciones, **x** representa la Cota. El ajuste se realizó restando **{cota_min}**.")

        # Iteramos grados 1 a 6
        for grado in range(1, 7):
            try:
                # 1. Ajuste usando la variable desplazada (X - Xmin)
                coeficientes = np.polyfit(x_shift, y_real, grado)
                model = np.poly1d(coeficientes)
                
                # 2. Predicciones y R^2
                y_pred_stats = model(x_shift) # Predicción sobre puntos conocidos para R2
                r2 = r2_score(y_real, y_pred_stats)
                
                # 3. Curva para gráfico
                y_curve = model(x_range_shift)
                fig.add_trace(go.Scatter(
                    x=x_range, y=y_curve, mode='lines', 
                    name=f'G{grado} (R²={r2:.4f})',
                    visible='legendonly' if grado > 3 else True
                ))

                # 4. Construcción del String solicitado
                # Formato: y = (Coef * (x - min)^n) + ...
                terms = []
                for i, c in enumerate(coeficientes):
                    potencia = grado - i
                    bloque = ""
                    
                    if potencia > 0:
                        bloque = f"({c:.12f} * (x - {cota_min})^ {potencia})"
                    else:
                        # Término independiente
                        bloque = f"({c:.12f})"
                    
                    terms.append(bloque)
                
                ecuacion_str = " + ".join(terms)
                
                # --- MOSTRAR RESULTADO ---
                # Usamos st.text para que sea fácil de copiar y pegar tal cual pediste
                header_str = f"Grado {grado} | R² = {r2:.8f}"
                body_str = f"y = {ecuacion_str}"
                
                with st.expander(header_str, expanded=True):
                    st.code(f"{header_str} | {body_str}", language="text")
                    # Versión LaTeX para informes visuales
                    st.latex(f"R^2 = {r2:.4f}")

            except Exception as e:
                st.error(f"Error en Grado {grado}: {e}")

        # Mostrar Gráfico
        fig.update_layout(
            title="Curvas Cota-Volumen (Modelos)",
            xaxis_title="Cota (m.s.n.m)", yaxis_title="Volumen (m³)",
            height=800, template="plotly_white", hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Carga datos primero.")

st.divider()

# --- SECCIÓN 3: INTERPOLACIÓN LINEAL ---
st.header("3. Interpoladora Lineal")
col1, col2 = st.columns(2)
with col1:
    try:
        cota_input = st.number_input("Cota a buscar:", value=cota_min, format="%.3f")
    except:
        cota_input = st.number_input("Cota a buscar:", value=0.00, format="%.3f")
with col2:
    if df is not None:
        val = np.interp(cota_input, df['Cota'], df['Volumen'])
        st.metric("Volumen Interpolado", f"{val:,.3f}")
    else:
        st.write("Esperando datos...")