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
    /* Estilo para resaltar la mejor ecuación */
    .mejor-ecuacion {
        border: 2px solid #00CC96;
        border-radius: 5px;
        padding: 10px;
        background-color: rgba(0, 204, 150, 0.1);
    }
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
        df_all = pd.read_csv(FILE_PATH)
        req_cols = ['ID_Poza', 'Cota', 'Volumen']
        if not all(col in df_all.columns for col in req_cols):
            st.error(f"⚠️ El CSV debe tener las columnas: {req_cols}")
            st.stop()

        lista_pozas = df_all['ID_Poza'].unique()
        poza_seleccionada = st.sidebar.selectbox("Seleccionar Poza:", lista_pozas)
        
        df_filtrado = df_all[df_all['ID_Poza'] == poza_seleccionada].copy()
        df_filtrado = df_filtrado.sort_values(by='Cota')
        
        cota_min_ref = df_filtrado['Cota'].min()
        cota_max_ref = df_filtrado['Cota'].max()
        
        df_filtrado['X_Shift'] = df_filtrado['Cota'] - cota_min_ref

        st.sidebar.success(f"✅ Datos cargados: {len(df_filtrado)} registros")
        st.sidebar.markdown(f"**Rango Cota:** {cota_min_ref} - {cota_max_ref} m")
        st.sidebar.dataframe(df_filtrado[['Cota', 'Volumen']], height=250, use_container_width=True)

    except Exception as e:
        st.error(f"Error leyendo el archivo: {e}")
        st.stop()
else:
    st.warning(f"⚠️ No se encuentra el archivo de datos.")
    st.stop()

# --- 2. LÓGICA PRINCIPAL ---
if df_filtrado is not None:
    
    x_real = df_filtrado['Cota'].values
    x_shift = df_filtrado['X_Shift'].values
    y_real = df_filtrado['Volumen'].values

    # --- A. GRÁFICO ---
    st.subheader(f"📊 Curva Característica: {poza_seleccionada}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_real, y=y_real, mode='markers', name='Datos Reales',
                             marker=dict(size=10, color='#00CC96', line=dict(width=1, color='black'))))
    
    x_range = np.linspace(cota_min_ref, cota_max_ref, 200)
    x_range_shift = x_range - cota_min_ref 
    
    resultados_data = [] 

    # Calcular Polinomios (Grado 1 a 6)
    for grado in range(1, 7):
        try:
            coefs = np.polyfit(x_shift, y_real, grado)
            model = np.poly1d(coefs)
            
            y_pred = model(x_shift)
            r2 = r2_score(y_real, y_pred)
            
            # --- CÁLCULO DE VARIACIÓN EN PORCENTAJE ---
            # Evitamos división por cero usando numpy errstate
            with np.errstate(divide='ignore', invalid='ignore'):
                variaciones_pct = np.abs((y_real - y_pred) / y_real) * 100
                # Si volumen real es 0, la variación suele ser infinita, la convertimos a 0 o NaN
                variaciones_pct = np.nan_to_num(variaciones_pct, nan=0.0, posinf=0.0, neginf=0.0)

            # Promedio omitiendo los primeros 5 datos (si existen)
            if len(variaciones_pct) > 5:
                promedio_var_pct = np.mean(variaciones_pct[5:])
            else:
                promedio_var_pct = np.mean(variaciones_pct)

            # Gráfico
            y_curve = model(x_range_shift)
            fig.add_trace(go.Scatter(
                x=x_range, y=y_curve, mode='lines', 
                name=f'G{grado} (Var {promedio_var_pct:.2f}%)',
                visible='legendonly' if grado > 3 else True
            ))
            
            # String Ecuación
            terms = []
            for i, c in enumerate(coefs):
                potencia = grado - i
                signo = "+" if c >= 0 else "-"
                val_abs = abs(c)
                if i == 0:
                    term_str = f"({c:.8f} * (x - {cota_min_ref})^{potencia})"
                else:
                    if potencia > 0:
                        term_str = f"{signo} ({val_abs:.8f} * (x - {cota_min_ref})^{potencia})"
                    else:
                        term_str = f"{signo} ({val_abs:.8f})"
                terms.append(term_str)
            
            ec_str = " ".join(terms)
            if ec_str.startswith("+ "): ec_str = ec_str[2:]
            
            resultados_data.append({
                'grado': grado,
                'r2': r2,
                'ecuacion': ec_str,
                'modelo': model,
                'variacion_pct': promedio_var_pct,
                'y_pred': y_pred
            })
            
        except:
            pass

    fig.update_layout(height=500, template="plotly_white", 
                      xaxis_title="Cota (m.s.n.m)", yaxis_title="Volumen (m³)",
                      hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- B. ECUACIONES (EXPANDIDAS) ---
    st.subheader("📐 Ecuaciones Generadas")
    st.caption(f"Nota: Variación promedio calculada ignorando los primeros 5 registros.")

    # Identificar la MEJOR ecuación (Menor variación %)
    mejor_modelo_idx = min(range(len(resultados_data)), key=lambda i: resultados_data[i]['variacion_pct'])
    mejor_modelo_data = resultados_data[mejor_modelo_idx]

    col_eq1, col_eq2 = st.columns(2)

    for i, item in enumerate(resultados_data):
        target_col = col_eq1 if i % 2 == 0 else col_eq2
        es_mejor = (i == mejor_modelo_idx)
        
        titulo = f"Grado {item['grado']} | R²={item['r2']:.4f} | Var={item['variacion_pct']:.2f}%"
        if es_mejor: titulo = "⭐ RECOMENDADA: " + titulo

        with target_col:
            # Petición 2: Todas expandidas (expanded=True)
            with st.expander(titulo, expanded=True):
                st.code(f"y = {item['ecuacion']}", language="text")

    st.divider()

    # --- CONTROL CENTRALIZADO DE MODELO (Petición 4 y 6) ---
    st.info("👇 **Selecciona aquí** la ecuación que deseas usar para la tabla comparativa y los cálculos de campo.")
    
    # Crear opciones para el selectbox
    opciones_grados = [d['grado'] for d in resultados_data]
    
    # El selectbox
    grado_seleccionado = st.selectbox(
        "Ecuación a utilizar:", 
        options=opciones_grados,
        index=opciones_grados.index(mejor_modelo_data['grado']), # Por defecto la mejor
        format_func=lambda x: f"Polinomio Grado {x}"
    )

    # Recuperar datos del modelo seleccionado
    modelo_activo = next(item for item in resultados_data if item['grado'] == grado_seleccionado)

    st.divider()

    # --- C. ANÁLISIS DE PRECISIÓN (Petición 3) ---
    st.header(f"4. Comparativa: Real vs Calculado (Grado {grado_seleccionado})")
    
    # Calcular variación porcentual fila por fila para la tabla
    with np.errstate(divide='ignore', invalid='ignore'):
        var_fila_pct = np.abs((y_real - modelo_activo['y_pred']) / y_real) * 100
        var_fila_pct = np.nan_to_num(var_fila_pct, nan=0.0)

    df_comparativo = pd.DataFrame({
        'Cota': x_real,
        'Volumen Real (m³)': y_real,
        'Volumen Calc (m³)': modelo_activo['y_pred'],
        'Variación (%)': var_fila_pct
    })
    
    # Formateo visual (Petición 3: Variación en %)
    st.dataframe(
        df_comparativo.style.format({
            'Cota': "{:.2f}",
            'Volumen Real (m³)': "{:,.2f}",
            'Volumen Calc (m³)': "{:,.2f}",
            'Variación (%)': "{:.2f}%"
        }).background_gradient(subset=['Variación (%)'], cmap="Reds"), 
        use_container_width=True
    )

    st.divider()

    # --- D. INTERPOLADORA ORIGINAL (Petición 5 - Intacta) ---
    st.subheader("🧮 Interpoladora Lineal Simple")
    col_cal1, col_cal2 = st.columns([1, 2])
    with col_cal1:
        cota_input = st.number_input("Ingresar Cota:", min_value=float(cota_min_ref)-10, max_value=float(cota_max_ref)+10, value=float(cota_min_ref), format="%.3f")
    with col_cal2:
        vol_interp = np.interp(cota_input, x_real, y_real)
        st.metric("Volumen Interpolado", f"{vol_interp:,.3f} m³")

    st.divider()

    # --- E. CALCULADORA DE CAMPO (Petición 6) ---
    st.header(f"5. Estimación de Porcentaje de Llenado (Usando Grado {grado_seleccionado})")
    
    col_f1, col_f2, col_f3 = st.columns(3)

    # Modelo matemático activo
    func_modelo = modelo_activo['modelo']

    with col_f1:
        st.markdown("**1. Definir 100% Operativo**")
        cota_100_input = st.number_input(
            "Cota al 100% (Campo):", 
            value=float(cota_max_ref),
            format="%.3f"
        )
        vol_100_calc = func_modelo(cota_100_input - cota_min_ref)
        st.metric("Volumen Total (100%)", f"{vol_100_calc:,.2f} m³")

    with col_f2:
        st.markdown("**2. Medición Actual**")
        cota_actual_input = st.number_input(
            "Cota Lectura Actual:", 
            value=float(cota_min_ref) + 1.0,
            format="%.3f"
        )
        vol_actual_calc = func_modelo(cota_actual_input - cota_min_ref)
        if vol_actual_calc < 0: vol_actual_calc = 0
        st.metric("Volumen Actual", f"{vol_actual_calc:,.2f} m³")

    with col_f3:
        st.markdown("**3. Resultado**")
        if vol_100_calc > 0:
            porcentaje_llenado = (vol_actual_calc / vol_100_calc) * 100
        else:
            porcentaje_llenado = 0
        
        st.metric("Porcentaje de Llenado", f"{porcentaje_llenado:.2f} %")
        st.progress(min(max(porcentaje_llenado/100, 0.0), 1.0))