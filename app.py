import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import zipfile
import subprocess
import requests
from typing import Dict, List
import time

# Intentar importar librerías de IA, si no están, se desactivará esa parte
try:
    from langchain_community.chat_models import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import HumanMessage, AIMessage
    IA_AVAILABLE = True
except ImportError:
    IA_AVAILABLE = False

# ==========================================
# CONFIGURACIÓN PRE-PROCESADO
# ==========================================
MODELO_OLLAMA = "llama3.1"  # Ajustable

CATEGORIAS_POSIBLES = [
    "Supermercado", "Restaurantes/Cafe", "Transporte", 
    "Suscripciones", "Salud/Farmacia", "Shopping", 
    "Servicios/Casa", "Inversión/Ahorro", "Trabajo/Oficina", "Otros"
]

# Configuración de la página
st.set_page_config(
    page_title="📊 Mi Dashboard Financiero",
    page_icon="💰",
    layout="wide"
)

# Inicializar session state
if 'df_transacciones' not in st.session_state:
    st.session_state.df_transacciones = pd.DataFrame()
if 'archivo_cargado' not in st.session_state:
    st.session_state.archivo_cargado = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False

# Columnas base definidas por el usuario (DF Procesado)
COLUMNAS_REQUERIDAS = [
    'Fecha', 'Comercio', 'Descripción Original', 'Importe', 'Categoría'
]

def parsear_fecha_flexible(fecha_valor):
    """
    Función para parsear fechas de manera más flexible
    Soporta múltiples formatos incluyendo timestamps
    """
    if pd.isna(fecha_valor):
        return None
    
    # Si ya es datetime, convertir a date
    if isinstance(fecha_valor, pd.Timestamp) or isinstance(fecha_valor, datetime):
        return fecha_valor
    
    # Si es un número (int/float)
    if isinstance(fecha_valor, (int, float)):
        # Verificar si parece formato AAAAMMDD (ej: 20240115)
        if 19000000 <= fecha_valor <= 21001231:
            try:
                return datetime.strptime(str(int(fecha_valor)), '%Y%m%d')
            except:
                pass
        
        # Timestamp de Excel
        try:
            if fecha_valor > 25569:
                return pd.to_datetime(fecha_valor, origin='1899-12-30', unit='D')
        except:
            pass
    
    # Si es string
    if isinstance(fecha_valor, str):
        fecha_str = str(fecha_valor).strip()
        
        # Lista de formatos extendida
        formatos = [
            '%Y-%m-%d %H:%M:%S', # 2024-07-08 10:19:55
            '%Y-%m-%d',          # 2024-07-08
            '%Y%m%d',            # 20240115
            '%d/%m/%Y',          # 15/01/2024
            '%d-%m-%Y',          # 15-01-2024
            '%Y/%m/%d',          # 2024/01/15
            '%d.%m.%Y',          # 15.01.2024
        ]
        
        for formato in formatos:
            try:
                return datetime.strptime(fecha_str, formato)
            except:
                continue
        
        # Fallback a pandas
        try:
            return pd.to_datetime(fecha_str, dayfirst=True, errors='coerce')
        except:
            return None
    
    return None

def crear_plantilla_csv():
    """Crear plantilla ZIP con CSVs para descargar usando las nuevas columnas y separador ;"""
    # Transacciones ejemplo
    datos_ejemplo = {
        'Fecha': ['2024-01-15 10:00:00', '2024-01-16 14:30:00', '2024-01-30 09:00:00'],
        'Comercio': ['Mercadona', 'Repsol', 'Empresa S.L.'],
        'Descripción Original': ['COMPRA TARJETA ...', 'PAGO GASOLINERA ...', 'NOMINA ENERO'],
        'Importe': ['-150,50', '-80,00', '2500,00'], # Ejemplo con coma decimal
        'Categoría': ['Alimentación', 'Transporte', 'Ingresos']
    }
    df_transacciones = pd.DataFrame(datos_ejemplo)
    
    # Metas ejemplo
    df_metas = pd.DataFrame({
        'Nombre_Meta': ['Ahorro Coche', 'Vacaciones'],
        'Monto_Objetivo': [5000, 2000],
        'Fecha_Limite': ['20241231', '20240630'],
        'Fecha_Creacion': ['20240101', '20240101']
    })
    
    # Instrucciones
    instrucciones = """INSTRUCCIONES DE USO
================================
1. FORMATO DE ARCHIVO: CSV delimitado por punto y coma (;)
   
2. FORMATO DE FECHA: Acepta YYYY-MM-DD HH:MM:SS o AAAAMMDD
   
3. FORMATO NUMÉRICO: Usa coma (,) para decimales (ej: 18,50)

4. COLUMNAS REQUERIDAS (Se mapean automáticamente):
   - fecha (o Fecha)
   - comercio (o Comercio)
   - concepto_raw (o Descripción Original)
   - importe (o Importe)
   - Categoria (o Categoría)

5. ARCHIVOS:
   - transacciones.csv
   - metas.csv
"""

    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Usar sep=';'
        zf.writestr('transacciones.csv', df_transacciones.to_csv(index=False, sep=';'))
        zf.writestr('metas.csv', df_metas.to_csv(index=False, sep=';'))
        zf.writestr('LEEME.txt', instrucciones)
    
    return output.getvalue()

    return output.getvalue()

# ==========================================
# FUNCIONES DE PRE-PROCESADO (IA + REGLAS)
# ==========================================
def is_ollama_running():
    try:
        # Check simple endpoint
        requests.get("http://localhost:11434", timeout=1)
        return True
    except:
        return False

def ensure_ollama_running():
    """Verifica si Ollama está corriendo, si no, lo inicia."""
    if not IA_AVAILABLE: return False
    
    if is_ollama_running():
        return True
        
    st.toast("🚀 Iniciando servidor Ollama...", icon="🤖")
    try:
        # Start ollama serve in background
        subprocess.Popen(["ollama", "serve"], shell=True)
        
        # Wait for up to 10 seconds
        for _ in range(10):
            time.sleep(1)
            if is_ollama_running():
                st.toast("✅ Ollama iniciado correctamente", icon="✅")
                return True
                
        st.error("❌ No pudimos iniciar Ollama autmáticamente. Ejecuta 'ollama serve' en tu terminal.")
        return False
    except Exception as e:
        st.error(f"Error lanzando Ollama: {e}")
        return False

def configurar_ia():
    """Configura el modelo de Ollama"""
    if not IA_AVAILABLE: return None
    
    # Asegurar que esté corriendo
    if not ensure_ollama_running():
        return None
        
    try:
        llm = ChatOllama(model=MODELO_OLLAMA, temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Eres un experto financiero. Clasifica el gasto basado en el comercio. "
                       "Responde EXCLUSIVAMENTE con una de estas categorías: {categorias}. "
                       "Si no estás seguro o es un nombre de persona, usa 'Otros'. "
                       "Si es comida o bebida, usa 'Restaurantes/Cafe'."),
            ("user", "Clasifica: '{concepto}'")
        ])
        return prompt | llm | StrOutputParser()
    except Exception as e:
        st.error(f"⚠️ Error conectando a Ollama: {e}")
        return None

def inicializar_pipeline(df):
    column_mapping = {
        'Tipo': 'tipo', 'Producto': 'cuenta', 'Fecha de inicio': 'fecha',
        'Descripción': 'concepto_raw', 'Importe': 'importe', 'Estado': 'estado'
    }
    # Solo renombra si existen
    df = df.rename(columns={k:v for k,v in column_mapping.items() if k in df.columns})
    
    # Conversión de fecha
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    
    # Limpieza importe
    if 'importe' in df.columns:
         if df['importe'].dtype == 'object':
            df['importe'] = df['importe'].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)
            
    return df

def limpiar_descripcion(row):
    texto = str(row.get('concepto_raw', ''))
    tipo = row.get('tipo', '')
    
    if 'Payment to ' in texto: texto = texto.replace('Payment to ', '')
    if 'To ' in texto and tipo == 'TRANSFER': return f"Transferencia: {texto.replace('To ', '')}"
    if 'Top-up' in texto: return 'Ingreso / Recarga'
        
    texto = texto.split('*')[0].split('  ')[0]
    return texto.strip().upper()

def categorizar_reglas(row):
    """Filtro rápido basado en palabras clave."""
    comercio = str(row.get('comercio', '')).upper()
    importe = row.get('importe', 0)
    
    # 1. Ingresos
    if importe > 0:
        return 'Ingresos' if any(x in comercio for x in ['NOMINA', 'TRANSFERENCIA']) else 'Otros Ingresos'

    # 2. Ruido (Movimientos internos)
    if any(x in comercio for x in ['PARA GUARDAR', 'APARTADO', 'REDONDEO', 'REVPOINTS', 'JESÚS MATÍAS', 'EXCHANGE', 'AHORRO']):
        return 'RUIDO' 

    # 3. Reglas Fijas
    if any(x in comercio for x in ['MERCADONA', 'CARREFOUR', 'LIDL', 'ALDI', 'CONSUM']): return 'Supermercado'
    if any(x in comercio for x in ['UBER', 'BOLT', 'RENFE', 'METRO', 'GASOLINERA', 'BP', 'REPSOL']): return 'Transporte'
    if any(x in comercio for x in ['AMZN', 'AMAZON', 'ZARA', 'PRIMARK', 'CORTE INGLES']): return 'Shopping'
    if any(x in comercio for x in ['NETFLIX', 'SPOTIFY', 'HBO', 'DISNEY']): return 'Suscripciones'
    
    return 'OTROS' 

def crear_backup_csv():
    """Crear ZIP con CSVs de datos actuales (separador ;)"""
    output = io.BytesIO()
    
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        if not st.session_state.df_transacciones.empty:
            df_export = st.session_state.df_transacciones.copy()
            # Eliminar columnas auxiliares de visualización si existen
            cols_to_drop = ['Año', 'Mes_Str']
            df_export = df_export.drop(columns=[c for c in cols_to_drop if c in df_export.columns])
            
            # Formatear fechas
            if 'Fecha' in df_export.columns:
                df_export['Fecha'] = df_export['Fecha'].apply(
                    lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) and hasattr(x, 'strftime') else x
                )
            # Formatear importes con coma decimal para exportación
            if 'Importe' in df_export.columns:
                df_export['Importe'] = df_export['Importe'].apply(lambda x: str(x).replace('.', ','))
                
            zf.writestr('transacciones.csv', df_export.to_csv(index=False, sep=';'))
        else:
            # Header vacío con las columnas correctas
            zf.writestr('transacciones.csv', ';'.join(COLUMNAS_REQUERIDAS) + '\n')
    
    return output.getvalue()

def procesar_archivos(uploaded_files):
    """Procesar archivos subidos (CSV o ZIP) asumiendo separador ; y decimales con coma"""
    df_todas_transacciones = pd.DataFrame()
    metas_cargadas = []
    
    archivos_map = {}
    
    try:
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
            
        for file in uploaded_files:
            if file.name.endswith('.zip'):
                with zipfile.ZipFile(file) as z:
                    for filename in z.namelist():
                        if filename.endswith('.csv') and not filename.startswith('__MACOSX'):
                            with z.open(filename) as f:
                                archivos_map[filename] = pd.read_csv(f, sep=';', decimal=',') # Default guess
            elif file.name.endswith('.csv'):
                # Leer primero para detectar formato: probar ; y luego ,
                try:
                    df_temp = pd.read_csv(file, sep=';', decimal=',')
                    if len(df_temp.columns) <= 1: # Si solo hay 1 columna, probablemente el SEPARADOR está mal
                        file.seek(0)
                        df_temp = pd.read_csv(file, sep=',')
                except:
                    file.seek(0)
                    df_temp = pd.read_csv(file, sep=',')
                archivos_map[file.name] = df_temp
        
        # PIPELINE DE PRE-PROCESADO INTELIGENTE
        # Detectar si hay archivos "crudos" y procesarlos
        for name, df in archivos_map.items():
            # Columnas típicas de raw (Revolut, bancos, etc)
            raw_cols = {'Tipo', 'Producto', 'Fecha de inicio', 'Importe'} # Ajustar según tu banco
            
            # Si parece un archivo crudo (tiene columnas raw)
            if raw_cols.intersection(df.columns):
                st.info(f"⚡ Detectado archivo crudo: {name}. Iniciando auto-procesado con reglas + IA...")
                
                # Barra de progreso
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 1. Pipeline Inicial
                status_text.text("Limpiando datos...")
                df_clean = inicializar_pipeline(df)
                df_clean['comercio'] = df_clean.apply(limpiar_descripcion, axis=1)
                
                # 2. Reglas
                progress_bar.progress(20)
                status_text.text("Aplicando reglas rápidas...")
                df_clean['Categoria'] = df_clean.apply(categorizar_reglas, axis=1)
                
                # 3. IA (Solo si hay 'OTROS')
                candidatos_ia = df_clean[
                    (df_clean['Categoria'] == 'OTROS') & 
                    (df_clean['importe'] < 0)
                ]['comercio'].unique()
                
                if len(candidatos_ia) > 0 and IA_AVAILABLE:
                    status_text.text(f"🤖 IA analizando {len(candidatos_ia)} comercios nuevos...")
                    chain = configurar_ia()
                    
                    if chain:
                        mapa_ia = {}
                        step = 0
                        total_ia = len(candidatos_ia)
                        
                        for comercio in candidatos_ia:
                            step += 1
                            progreso_ia = 20 + int((step / total_ia) * 70) # 20% a 90%
                            progress_bar.progress(min(progreso_ia, 90))
                            
                            try:
                                res = chain.invoke({
                                    "categorias": ", ".join(CATEGORIAS_POSIBLES),
                                    "concepto": comercio
                                })
                                cat_limpia = res.strip().replace('"', '').replace('.', '')
                                mapa_ia[comercio] = cat_limpia if any(c in cat_limpia for c in CATEGORIAS_POSIBLES) else 'Otros'
                            except:
                                mapa_ia[comercio] = 'Otros'
                        
                        # Aplicar cambios
                        def aplicar_ia(row):
                            if row['Categoria'] == 'OTROS' and row['importe'] < 0:
                                return mapa_ia.get(row['comercio'], 'Otros')
                            return row['Categoria']
                        
                        df_clean['Categoria'] = df_clean.apply(aplicar_ia, axis=1)
                elif len(candidatos_ia) > 0 and not IA_AVAILABLE:
                    st.warning("⚠️ Librerías de Langchain no encontradas. Saltando clasificación IA.")
                
                progress_bar.progress(100)
                status_text.text("✅ Procesado completado.")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
                
                # Filtrar RUIDO
                df_clean = df_clean[df_clean['Categoria'] != 'RUIDO'].copy()
                
                # Reemplazar el dataframe crudo por el procesado en el mapa
                archivos_map[name] = df_clean
        
        # Buscar transacciones (Lógica existente adaptada)
        df_trans = None
        for name, df in archivos_map.items():
            # Mapeo de columnas del usuario a internas
            df.columns = df.columns.str.strip()
            rename_map = {
                'fecha': 'Fecha',
                'comercio': 'Comercio',
                'concepto_raw': 'Descripción Original', 
                'importe': 'Importe',
                'Categoria': 'Categoría',
                'categoria': 'Categoría'
            }
            df = df.rename(columns=rename_map)

            # Identificar si tiene las columnas clave
            columnas_clave = {'Importe', 'Fecha', 'Categoría'}
            
            if 'transacciones' in name.lower() or columnas_clave.issubset(df.columns):
                df_trans = df
                break
        
        if df_trans is not None:
            # Limpiar datos
            df_trans = df_trans.dropna(how='all')
            
            # Procesar Importe (asegurar que es float)
            if 'Importe' in df_trans.columns:
                # Si se leyó como string por algún motivo, limpiar
                if df_trans['Importe'].dtype == 'object':
                    # Eliminar puntos de miles si existen y reemplazar coma decimal
                    # Asumimos formato europeo: 1.234,56 -> 1234.56
                    # O formato simple: 1234,56 -> 1234.56
                    
                    # Primero, si hay puntos y comas, asumimos punto = mil, coma = decimal
                    # Si solo hay comas, coma = decimal
                    
                    def limpiar_importe(val):
                        if isinstance(val, str):
                            val = val.strip()
                            if ',' in val and '.' in val:
                                val = val.replace('.', '').replace(',', '.')
                            elif ',' in val:
                                val = val.replace(',', '.')
                        return val

                    df_trans['Importe'] = df_trans['Importe'].apply(limpiar_importe)
                    df_trans['Importe'] = pd.to_numeric(df_trans['Importe'], errors='coerce')
            
            # Procesar fechas
            fechas_procesadas = []
            fechas_problematicas = []
            
            if 'Fecha' in df_trans.columns:
                for idx, val in enumerate(df_trans['Fecha']):
                    fecha_proc = parsear_fecha_flexible(val)
                    if fecha_proc is None:
                        fechas_problematicas.append(f"Fila {idx+2}: {val}")
                    fechas_procesadas.append(fecha_proc)
                
                df_trans['Fecha'] = fechas_procesadas
                
                if fechas_problematicas:
                    st.warning(f"⚠️ Se encontraron {len(fechas_problematicas)} fechas inválidas.")
                    with st.expander("Ver detalles"):
                        for fp in fechas_problematicas[:10]:
                            st.write(fp)
                
                # Eliminar filas sin fecha válida
                df_trans = df_trans.dropna(subset=['Fecha'])

                # Ordenar por Categoría y luego Fecha (más reciente primero)
                if 'Categoría' in df_trans.columns:
                    df_trans = df_trans.sort_values(by=['Categoría', 'Fecha'], ascending=[True, False])

                df_todas_transacciones = df_trans
            else:
                st.error("❌ El archivo no contiene la columna 'Fecha'")

        # Buscar metas
        df_metas = None
        for name, df in archivos_map.items():
            if 'metas' in name.lower() or {'Nombre_Meta', 'Monto_Objetivo'}.issubset(df.columns):
                df_metas = df
                break
        
        if df_metas is not None:
            for _, row in df_metas.iterrows():
                if pd.notna(row.get('Nombre_Meta')) and pd.notna(row.get('Monto_Objetivo')):
                    meta = {
                        'nombre': str(row['Nombre_Meta']),
                        'monto': float(row['Monto_Objetivo']),
                        'fecha_creacion': datetime.now().date()
                    }
                    
                    if 'Fecha_Creacion' in row:
                        fc = parsear_fecha_flexible(row['Fecha_Creacion'])
                        if fc: meta['fecha_creacion'] = fc.date()
                        
                    if 'Fecha_Limite' in row:
                        fl = parsear_fecha_flexible(row['Fecha_Limite'])
                        if fl: meta['fecha_limite'] = fl.date()
                    else:
                        meta['fecha_limite'] = None
                        
                    metas_cargadas.append(meta)
                    
                if not df.empty or metas_cargadas: # Mantener compatibilidad simple
                    df_todas_transacciones = df
                    break

        return df_todas_transacciones, []


    except Exception as e:
        st.error(f"Error procesando archivos: {e}")
        return None, []

def calcular_insights(df):
    """Calcular insights financieros avanzados"""
    insights = []
    
    if df.empty or 'Importe' not in df.columns:
        return []
        
    try:
        # Separar ingresos y gastos
        ingresos_df = df[df['Importe'] > 0]
        gastos_df = df[df['Importe'] < 0].copy()
        gastos_df['Importe_Abs'] = gastos_df['Importe'].abs()
        
        total_ingresos = ingresos_df['Importe'].sum()
        total_gastos = gastos_df['Importe_Abs'].sum()
        balance = total_ingresos - total_gastos
        
        # 1. Tasa de Ahorro
        tasa_ahorro = (balance / total_ingresos * 100) if total_ingresos > 0 else 0
        icono_ahorro = "🟢" if tasa_ahorro > 20 else "🟡" if tasa_ahorro > 0 else "🔴"
        insights.append({
            "titulo": "Tasa de Ahorro",
            "valor": f"{tasa_ahorro:.1f}%",
            "desc": f"{icono_ahorro} De tus ingresos totales ({total_ingresos:,.0f}€)",
            "tipo": "metric"
        })

        # 2. Promedio Diario Real
        if not gastos_df.empty:
            dias_con_gastos = gastos_df['Fecha'].dt.date.nunique()
            rango_dias = (gastos_df['Fecha'].max() - gastos_df['Fecha'].min()).days + 1
            gasto_diario_real = total_gastos / max(1, rango_dias)
            insights.append({
                "titulo": "Gasto Diario Promedio",
                "valor": f"${gasto_diario_real:,.2f}",
                "desc": f"En un periodo de {rango_dias} días",
                "tipo": "metric"
            })
            
        # 3. Ticket Promedio (Gasto promedio por compra)
        if not gastos_df.empty:
            ticket_promedio = gastos_df['Importe_Abs'].mean()
            insights.append({
                "titulo": "Ticket Promedio",
                "valor": f"${ticket_promedio:,.2f}",
                "desc": "Valor promedio de cada compra",
                "tipo": "metric"
            })
            
        # 4. Día más costoso de la semana
        if not gastos_df.empty:
            gastos_df['Dia_Semana'] = gastos_df['Fecha'].dt.day_name(locale='es_ES') # Requiere locale, fallback a inglés si falla o usar números
            # Usaremos mapa manual para asegurar español
            dias_map = {0:'Lunes', 1:'Martes', 2:'Miércoles', 3:'Jueves', 4:'Viernes', 5:'Sábado', 6:'Domingo'}
            gastos_df['Dia_Num'] = gastos_df['Fecha'].dt.dayofweek
            gastos_df['Dia_Nombre'] = gastos_df['Dia_Num'].map(dias_map)
            
            dia_mas_costoso = gastos_df.groupby('Dia_Nombre')['Importe_Abs'].mean().idxmax()
            monto_dia = gastos_df.groupby('Dia_Nombre')['Importe_Abs'].mean().max()
            
            insights.append({
                "titulo": "Día 'Peligroso'",
                "valor": dia_mas_costoso,
                "desc": f"Promedio de gasto: ${monto_dia:,.0f}",
                "tipo": "highlight"
            })
            
        # 5. Top Categoría
        if not gastos_df.empty and 'Categoría' in gastos_df.columns:
            top_cat = gastos_df.groupby('Categoría')['Importe_Abs'].sum().idxmax()
            pct_cat = (gastos_df.groupby('Categoría')['Importe_Abs'].sum().max() / total_gastos) * 100
            insights.append({
                "titulo": "Categoría Principal",
                "valor": top_cat,
                "desc": f"Representa el {pct_cat:.1f}% de tus gastos",
                "tipo": "highlight"
            })

    except Exception as e:
        insights.append({"titulo": "Error", "valor": "Error cálculo", "desc": str(e), "tipo": "error"})
        
    return insights

def generate_financial_context(df):
    """Genera un resumen textual de los datos para la IA"""
    if df.empty: return "No hay datos cargados aún."
    
    total_ingresos = df[df['Importe'] > 0]['Importe'].sum()
    total_gastos = df[df['Importe'] < 0]['Importe'].abs().sum()
    balance = total_ingresos - total_gastos
    
    top_gastos = df[df['Importe'] < 0].groupby('Categoría')['Importe'].sum().abs().nlargest(5)
    top_gastos_str = "\n".join([f"- {cat}: ${val:,.2f}" for cat, val in top_gastos.items()])
    
    context = f"""
    ESTADO FINANCIERO ACTUAL:
    - Balance: ${balance:,.2f}
    - Ingresos Totales: ${total_ingresos:,.2f}
    - Gastos Totales: ${total_gastos:,.2f}
    
    TOP 5 CATEGORÍAS DE GASTO:
    {top_gastos_str}
    
    Rango de fechas: {df['Fecha'].min().date()} a {df['Fecha'].max().date()}
    """
    return context

def render_chatbot():
    """Renderiza el chatbot flotante"""
    if not IA_AVAILABLE: return

    # CSS para el botón flotante y la ventana de chat
    st.markdown("""
        <style>
        .floating-chat-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
            background-color: #FF4B4B;
            color: white;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 30px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            cursor: pointer;
            transition: transform 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .floating-chat-btn:hover {
            transform: scale(1.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Botón flotante (simulado con columnas para posicionamiento aproximado o sidebar, 
    # pero Streamlit nativo es limitado para 'fixed'. Usamos un expander en sidebar o un truco)
    # Mejor enfoque simple: Un expander en la sidebar que siempre está ahí, 
    # O un botón en la sidebar.
    
    # Vamos a usar la sidebar para contener el chat "flotante" de forma más nativa
    with st.sidebar:
        st.markdown("---")
        if st.checkbox("💬 Asistente IA", value=st.session_state.chat_open, key='toggle_chat'):
            st.session_state.chat_open = True
            
            st.subheader("🤖 FinanzasBot")
            
            # Contenedor de mensajes
            chat_container = st.container()
            with chat_container:
                # Mostrar últimos 5 mensajes para no saturar
                for msg in st.session_state.chat_history[-10:]:
                    if isinstance(msg, HumanMessage):
                        st.info(f"� {msg.content}")
                    elif isinstance(msg, AIMessage):
                        st.success(f"🤖 {msg.content}")
            
            # Input
            user_input = st.text_input("Pregunta algo sobre tus gastos...", key="chat_input")
            
            if st.button("Enviar", key="send_chat"):
                if user_input:
                    # Añadir usuario
                    st.session_state.chat_history.append(HumanMessage(content=user_input))
                    
                    # Generar respuesta
                    context = generate_financial_context(st.session_state.df_transacciones)
                    chain = configurar_ia()
                    
                    if chain:
                        with st.spinner("Pensando..."):
                            try:
                                # Prompt modificado para incluir contexto
                                full_prompt = {
                                    "categorias": "", # No usado en este modo conversación
                                    "concepto": user_input # Reusamos variable
                                }
                                # Creamos un prompt ad-hoc para chat
                                chat_llm = ChatOllama(model=MODELO_OLLAMA, temperature=0.7)
                                system_msg = (
                                    "Eres un asistente financiero personal amable y perspicaz. "
                                    "Usa el siguiente contexto de las finanzas del usuario para responder preguntas. "
                                    "Sé conciso y directo. Si no sabes algo, dilo. "
                                    "No inventes datos numéricos que no estén en el contexto.\n\n"
                                    f"CONTEXTO:\n{context}"
                                )
                                messages = [
                                    ("system", system_msg),
                                    *[(("user" if isinstance(m, HumanMessage) else "assistant"), m.content) 
                                      for m in st.session_state.chat_history[-5:]] # Historial corto
                                ]
                                
                                prompt_template = ChatPromptTemplate.from_messages(messages)
                                chat_chain = prompt_template | chat_llm | StrOutputParser()
                                
                                response = chat_chain.invoke({})
                                st.session_state.chat_history.append(AIMessage(content=response))
                                st.rerun() # Refrescar para mostrar mensaje
                            except Exception as e:
                                st.error(f"Error IA: {e}")
        else:
            st.session_state.chat_open = False

def main():
    st.title("�📊 Dashboard Financiero (AI Processed)")
    st.markdown("---")
    
    st.sidebar.title("🧭 Navegación")
    pagina = st.sidebar.radio(
        "Sección:",
        ["📥 Cargar Datos", "📊 Dashboard", "💡 Insights", "� Asistente IA", "�💾 Descargar"]
    )
    
    if pagina == "📥 Cargar Datos":
        st.header("📥 Cargar Datos (CSV)")
        st.info("Formato de fecha esperado: AAAAMMDD o YYYY-MM-DD. Separador: Punto y coma (;)")
        
        st.subheader("Subir Datos")
        files = st.file_uploader("Sube CSV o ZIP", type=['csv', 'zip'], accept_multiple_files=True)
        if files:
            df, _ = procesar_archivos(files)
            if not df.empty:
                st.session_state.df_transacciones = df
                st.session_state.archivo_cargado = True
                st.success(f"✅ Cargados: {len(df)} transacciones")

    elif pagina == "📊 Dashboard":
        st.header("📊 Dashboard Financiero")
        df = st.session_state.df_transacciones
        
        if df.empty:
            st.warning("⚠️ Carga datos primero")
            return

        # Preprocesamiento de fechas para filtros
        df['Año'] = df['Fecha'].dt.year
        df['Mes_Str'] = df['Fecha'].dt.strftime('%Y-%m')
        
        # --- FILTROS AVANZADOS ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Filtros Avanzados")
        
        # 1. Filtro de Fechas (Rango)
        min_date = df['Fecha'].min().date()
        max_date = df['Fecha'].max().date()
        
        fechas_sel = st.sidebar.date_input(
            "Seleccionar Rango de Fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Selecciona una fecha de inicio y fin para filtrar los datos."
        )
        
        # Aplicar filtro de fecha
        if isinstance(fechas_sel, tuple) and len(fechas_sel) == 2:
            start_date, end_date = fechas_sel
            df_filtered = df[
                (df['Fecha'].dt.date >= start_date) & 
                (df['Fecha'].dt.date <= end_date)
            ].copy()
        else:
            df_filtered = df.copy()

        # 2. Filtro de Años (Opcional, actúa sobre el rango seleccionado)
        años_disponibles = sorted(df_filtered['Año'].unique(), reverse=True)
        if años_disponibles:
            años_sel = st.sidebar.multiselect("Refinar por Año", años_disponibles, default=años_disponibles)
            if años_sel:
                df_filtered = df_filtered[df_filtered['Año'].isin(años_sel)]

        # 3. Filtro de Categorías
        if 'Categoría' in df_filtered.columns:
            cats_disponibles = sorted(df_filtered['Categoría'].unique())
            cats_sel = st.sidebar.multiselect("Refinar por Categoría", cats_disponibles, default=cats_disponibles)
            if cats_sel:
                df_filtered = df_filtered[df_filtered['Categoría'].isin(cats_sel)]
        
        # Preparación de datos globales
        df_gastos = df_filtered[df_filtered['Importe'] < 0].copy()
        df_gastos['Importe_Abs'] = df_gastos['Importe'].abs()
        
        df_ingresos = df_filtered[df_filtered['Importe'] > 0].copy()

        # --- KPIs (usando df filtrado) ---
        # --- KPIs con Comparativa (MoM) ---
        ingresos = df_filtered[df_filtered['Importe'] > 0]['Importe'].sum()
        gastos = abs(df_filtered[df_filtered['Importe'] < 0]['Importe'].sum())
        balance = ingresos - gastos
        
        # Calcular mes anterior para comparativa
        try:
             # Asumiendo que el filtro es por rango o total. Si es total, comparamos últimos 30 días vs anteriores?
             # O mejor, si hay filtro de mes, comparamos con mes anterior.
             # Simplificación: Comparamos este mes actual (según datos) vs mes previo en los datos globales
             
            fecha_max = df['Fecha'].max()
            mes_actual = fecha_max.month
            anio_actual = fecha_max.year
            
            # Definir "Periodo Actual" como el mes más reciente de los datos CASI SIEMPRE, 
            # pero aquí el usuario puede haber filtrado.
            # Vamos a calcular variaciones basadas en la selección actual vs todo el dataset si es posible, idealmente
            # deberíamos detectar si el filtro es un mes específico.
            
            # Lógica simple: Calcular KPI global del filtro seleccionado
            # Y mostrar variaciones visuales simuladas o calculadas si se selecciona 1 mes.
            
            # Mejor aproximación: Comparativa del mes más reciente en el FILTRO vs el mes anterior a ese en los DATOS GLOBALES
            
            ultimo_mes_filtro = df_filtered['Fecha'].max().month
            ultimo_anio_filtro = df_filtered['Fecha'].max().year
            
            # Datos mes anterior
            fecha_referencia = datetime(ultimo_anio_filtro, ultimo_mes_filtro, 1) - timedelta(days=1)
            mes_previo = fecha_referencia.month
            anio_previo = fecha_referencia.year
            
            df_previo = df[
                (df['Fecha'].dt.month == mes_previo) & 
                (df['Fecha'].dt.year == anio_previo)
            ]
            
            # Gastos mes previo
            gastos_previo = abs(df_previo[df_previo['Importe'] < 0]['Importe'].sum())
            ingresos_previo = df_previo[df_previo['Importe'] > 0]['Importe'].sum()
            
            delta_gastos = gastos - gastos_previo
            delta_ingresos = ingresos - ingresos_previo
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Balance", f"${balance:,.2f}")
            c2.metric("Ingresos", f"${ingresos:,.2f}", delta=f"{delta_ingresos:,.2f} vs mes ant.")
            c3.metric("Gastos", f"${gastos:,.2f}", delta=f"{-delta_gastos:,.2f} vs mes ant.", delta_color="inverse")
            c4.metric("Registros", len(df_filtered))
            
        except:
             # Fallback si falla cálculo de fechas
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Balance", f"${balance:,.2f}")
            c2.metric("Ingresos", f"${ingresos:,.2f}")
            c3.metric("Gastos", f"${gastos:,.2f}")
            c4.metric("Registros", len(df_filtered))
        
        st.markdown("---")
        
        # --- GRÁFICOS ---
        
        # --- GRÁFICOS AVANZADOS ---

        # 1. Resumen Mensual: Ingresos vs Gastos vs Balance
        st.subheader("�️ Balance Mensual")
        
        # Agrupar por mes
        resumen_mensual = df_filtered.groupby(['Mes_Str']).agg({
            'Importe': lambda x: x[x>0].sum(), # Ingresos
        }).rename(columns={'Importe': 'Ingresos'})
        
        resumen_mensual['Gastos'] = df_filtered.groupby(['Mes_Str'])['Importe'].apply(lambda x: abs(x[x<0].sum()))
        resumen_mensual['Balance'] = df_filtered.groupby(['Mes_Str'])['Importe'].sum()
        resumen_mensual = resumen_mensual.reset_index()

        fig_balance = go.Figure()
        
        # Barras de Ingresos
        fig_balance.add_trace(go.Bar(
            x=resumen_mensual['Mes_Str'],
            y=resumen_mensual['Ingresos'],
            name='Ingresos',
            marker_color='#2ecc71'
        ))
        
        # Barras de Gastos
        fig_balance.add_trace(go.Bar(
            x=resumen_mensual['Mes_Str'],
            y=resumen_mensual['Gastos'],
            name='Gastos',
            marker_color='#e74c3c'
        ))
        
        # Línea de Balance
        fig_balance.add_trace(go.Scatter(
            x=resumen_mensual['Mes_Str'],
            y=resumen_mensual['Balance'],
            name='Balance Neto',
            line=dict(color='#3498db', width=3),
            mode='lines+markers'
        ))
        
        fig_balance.update_layout(
            barmode='group',
            xaxis_title="Mes",
            yaxis_title="Importe ($)",
            hovermode="x unified",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_balance, use_container_width=True)
        
        st.markdown("---")
        
        # 1.5. Análisis Temporal y Distribución (Nuevos Gráficos)
        st.subheader("📈 Variación Temporal y Distribución")
        
        st.markdown("##### 📈 Variación de Gastos por Categoría (Tiempo)")
        if not df_gastos.empty:
            # Agrupar por mes y categoría para línea/área
            gastos_mes_cat = df_gastos.groupby(
                [df_gastos['Fecha'].dt.to_period('M').astype(str).rename('Mes'), 'Categoría']
            )['Importe_Abs'].sum().reset_index()
            
            fig_area = px.area(
                gastos_mes_cat,
                x='Mes',
                y='Importe_Abs',
                color='Categoría',
                title="Evolución Acumulada de Gastos",
                labels={'Importe_Abs': 'Gasto ($)'},
                markers=True
            )
            st.plotly_chart(fig_area, use_container_width=True)
    
        st.markdown("---")
        
        st.markdown("##### 🥧 Distribución Total")
        if not df_gastos.empty:
            fig_pie = px.pie(
                df_gastos,
                values='Importe_Abs',
                names='Categoría',
                hole=0.4,
                title="Gastos por Categoría (%)"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")

        # 2. Análisis de Gastos (Dos columnas)
        st.subheader("📉 Análisis Detallado de Gastos")
        
        st.markdown("##### 📦 Mapa de Gastos (Treemap)")
        if not df_gastos.empty:
            # Treemap para ver jerarquía/tamaño relativo
            fig_tree = px.treemap(
                df_gastos,
                path=['Categoría', 'Comercio'],
                values='Importe_Abs',
                color='Categoría',
                color_discrete_sequence=px.colors.qualitative.Prism,
                title="Distribución Jerárquica de Gastos"
            )
            fig_tree.update_traces(textinfo="label+value+percent entry")
            st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.info("No hay gastos para mostrar.")

        st.markdown("---")

        st.markdown("##### 📊 Distribución por Categoría (Boxplot)")
        if not df_gastos.empty:
            # Boxplot para ver distribución y outliers
            fig_box = px.box(
                df_gastos,
                x='Categoría',
                y='Importe_Abs',
                color='Categoría',
                points="all", # Mostrar todos los puntos
                title="Variabilidad de Gastos por Categoría",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_box.update_layout(showlegend=False, yaxis_title="Importe Gasto ($)")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No hay gastos para mostrar.")

        # 3. Evolución Temporal Detallada
        st.markdown("### ⏳ Cronología de Transacciones")
        if not df_filtered.empty:
            df_scatter = df_filtered.copy()
            df_scatter['Tipo'] = df_scatter['Importe'].apply(lambda x: 'Ingreso' if x > 0 else 'Gasto')
            df_scatter['Tamaño'] = df_scatter['Importe'].abs()
            
            fig_scatter = px.scatter(
                df_scatter,
                x='Fecha',
                y='Importe',
                color='Categoría',
                size='Tamaño',
                hover_data=['Comercio', 'Descripción Original'],
                title="Transacciones en el Tiempo (Scatter Plot)",
                color_discrete_sequence=px.colors.qualitative.Prism,
                opacity=0.7
            )
            # Línea de cero
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")

        # 4. Tabla de Top Gastos
        st.markdown("### 🔝 Mayores Gastos")
        if not df_gastos.empty:
            top_gastos = df_gastos.nlargest(10, 'Importe_Abs')[
                ['Fecha', 'Comercio', 'Categoría', 'Importe', 'Descripción Original']
            ]
            st.dataframe(
                top_gastos.style.format({'Importe': "${:,.2f}"}),
                use_container_width=True,
                hide_index=True
            )

    elif pagina == " Insights":
        st.header("💡 Análisis de Hábitos Financieros")
        
        if not st.session_state.df_transacciones.empty:
            insights_data = calcular_insights(st.session_state.df_transacciones)
            
            # Dividir en métricas y destacados
            metrics = [i for i in insights_data if i.get('tipo') == 'metric']
            highlights = [i for i in insights_data if i.get('tipo') == 'highlight']
            
            # Mostrar Métricas en filas de 3
            if metrics:
                st.subheader("Indicadores Clave")
                cols = st.columns(len(metrics))
                for idx, m in enumerate(metrics):
                    cols[idx].metric(m['titulo'], m['valor'], m['desc'])
            
            st.markdown("---")
            
            # Mostrar Destacados
            if highlights:
                st.subheader("🔍 Patrones Detectados")
                c1, c2 = st.columns(2)
                for idx, h in enumerate(highlights):
                    with c1 if idx % 2 == 0 else c2:
                        st.info(f"**{h['titulo']}**: {h['valor']}\n\n_{h['desc']}_")
                        
            # Análisis Semanal Gráfico
            st.markdown("---")
            st.subheader("📅 Patrón de Gasto Semanal")
            df = st.session_state.df_transacciones
            gastos = df[df['Importe'] < 0].copy()
            gastos['Importe'] = gastos['Importe'].abs()
            gastos['Dia_Semana'] = gastos['Fecha'].dt.dayofweek
            dias_map = {0:'Lunes', 1:'Martes', 2:'Miércoles', 3:'Jueves', 4:'Viernes', 5:'Sábado', 6:'Domingo'}
            gastos['Dia_Nombre'] = gastos['Dia_Semana'].map(dias_map)
            
            # Ordenar correctamente
            gastos_sem = gastos.groupby('Dia_Nombre')['Importe'].mean().reindex(dias_map.values()).reset_index()
            
            fig_sem = px.bar(
                gastos_sem, x='Dia_Nombre', y='Importe',
                title="Gasto Promedio por Día de la Semana",
                labels={'Importe': 'Gasto Promedio ($)', 'Dia_Nombre': 'Día'},
                color='Importe', color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_sem, use_container_width=True)

        else:
            st.warning("Carga datos primero para ver el análisis.")

    elif pagina == "💬 Asistente IA":
        st.header("💬 Asistente Financiero IA")
        st.info("Pregunta lo que quieras sobre tus finanzas. La IA analizará tus datos cargados.")
        
        if not IA_AVAILABLE:
            st.error("❌ Langchain/Ollama no están disponibles.")
            st.warning("Asegúrate de tener instalado 'langchain-community' y Ollama corriendo.")
        else:
            # Asegurar que Ollama corre
            ensure_ollama_running()
            
            # Contenedor de chat
            chat_container = st.container()
            
            # Input fijo abajo
            prompt = st.chat_input("Escribe tu pregunta aquí...")
            
            if prompt:
                # Añadir mensaje usuario
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                
                # Generar contexto actualizado
                df = st.session_state.df_transacciones
                context = generate_financial_context(df)
                
                # Invocar IA
                chain = configurar_ia()
                if chain:
                    with st.spinner("Analizando tus datos..."):
                        try:
                           # Prompt modificado para incluir contexto
                            full_prompt = {
                                "categorias": "", # No usado en este modo conversación
                                "concepto": prompt 
                            }
                            # Creamos un prompt ad-hoc para chat
                            chat_llm = ChatOllama(model=MODELO_OLLAMA, temperature=0.7)
                            system_msg = (
                                "Eres un asistente financiero personal amable y perspicaz. "
                                "Usa el siguiente contexto de las finanzas del usuario para responder preguntas. "
                                "Sé conciso, directo y usa emojis. "
                                "No inventes datos numéricos que no estén en el contexto.\n\n"
                                f"CONTEXTO:\n{context}"
                            )
                            messages = [
                                ("system", system_msg),
                                *[(("user" if isinstance(m, HumanMessage) else "assistant"), m.content) 
                                  for m in st.session_state.chat_history[-10:]] # Historial
                            ]
                            
                            prompt_template = ChatPromptTemplate.from_messages(messages)
                            chat_chain = prompt_template | chat_llm | StrOutputParser()
                            
                            response = chat_chain.invoke({})
                            st.session_state.chat_history.append(AIMessage(content=response))
                            
                        except Exception as e:
                            st.error(f"Error IA: {e}")
            
            # Mostrar historial (lo mostramos al final para que el input quede abajo natural)
            with chat_container:
                for msg in st.session_state.chat_history:
                    if isinstance(msg, HumanMessage):
                        with st.chat_message("user"):
                            st.write(msg.content)
                    elif isinstance(msg, AIMessage):
                        with st.chat_message("assistant"):
                            st.write(msg.content)
            
            if st.button("🗑️ Borrar Historial"):
                st.session_state.chat_history = []
                st.rerun()

    elif pagina == "💾 Descargar":
        st.header("💾 Backup")
        if not st.session_state.df_transacciones.empty:
            st.download_button(
                "📥 Descargar Todo (ZIP)",
                crear_backup_csv(),
                f"backup_{datetime.now().strftime('%Y%m%d')}.zip",
                "application/zip"
            )

if __name__ == "__main__":
    main()
