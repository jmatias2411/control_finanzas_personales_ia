# 📊 Personal Finance AI Dashboard

> **Tu centro de mando financiero, 100% privado y local.**
> Una aplicación "Todo en Uno" que transforma tus extractos bancarios crudos en inteligencia financiera usando Reglas Heurísticas y LLMs Locales (Ollama).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Ollama](https://img.shields.io/badge/AI-Llama3-orange)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local-green)

## 🧐 ¿Qué es esto?

Este proyecto es una aplicación web minimalista diseñada para automatizar el control de gastos personales. A diferencia de las soluciones comerciales, **aquí tú eres el dueño de tus datos**.

**`app.py`** es el cerebro de la operación. Funciona como una solución unificada que:
1.  **Ingesta:** Permite subir tu CSV bancario crudo (ej. Revolut) directamente desde el navegador.
2.  **Procesa (ETL):** Limpia, normaliza y categoriza cada movimiento en tiempo real usando un motor híbrido (Regex para lo obvio + Llama 3.1 para lo complejo).
3.  **Visualiza:** Genera un dashboard interactivo instantáneo con tus KPIs financieros.

## 🚀 Características Clave

* **🔒 Privacidad Absoluta:** La IA (Ollama) corre localmente en tu máquina. Tus finanzas nunca tocan la nube.
* **🧠 IA Híbrida:** Combina la velocidad de las reglas fijas con la flexibilidad de un LLM para categorizar gastos ambiguos (ej: "Kiosco Pepe" → "Otros" vs "Restaurantes").
* **🧹 Smart Cleaning:** Filtra automáticamente el "ruido" financiero: transferencias internas, huchas y cambios de divisa.
* **📈 Analytics Visuales:** Gráficos de Sankey (flujo de dinero), Donut Charts y evolución mensual de ahorro.

## 🛠️ Stack Tecnológico

* **Core:** Python
* **Frontend & UI:** Streamlit
* **Motor de Datos:** Pandas
* **Inteligencia Artificial:** LangChain + Ollama (Modelo: `llama3.1`)
* **Gráficos:** Plotly Express

## ⚙️ Instalación y Despliegue

### 1. Prerrequisitos

Necesitas tener **Ollama** instalado y corriendo en tu máquina:

```bash
# 1. Instala Ollama desde ollama.com
# 2. Descarga el modelo ligero recomendado:
ollama pull llama3.1
````

### 2\. Clonar el repositorio

```bash
git clone https://github.com/jmatias2411/control_finanzas_personales_ia finanzas-ai
cd finanzas-ai
```

### 3\. Preparar el entorno

Es recomendable usar un entorno virtual. Además, generamos las dependencias necesarias:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instala las librerías necesarias
pip install streamlit pandas plotly langchain-community langchain-core ollama
```

### 4\. 🚀 Ejecutar la App

Una vez instalado todo, solo necesitas un comando:

```bash
streamlit run app.py
```

*Esto abrirá una pestaña en tu navegador. Arrastra tu CSV bancario y deja que la magia ocurra.*

## 📂 Estructura del Proyecto

El proyecto sigue la filosofía KISS (*Keep It Simple, Stupid*):

```text
📁 finanzas-ai/
├── 📄 app.py              # El Monolito: UI, Lógica ETL y Visualización
├── 📄 README.md           # Esta documentación
└── 📄 requirements.txt    # (Generar con: pip freeze > requirements.txt)
```

## 📝 Roadmap

  - [ ] Integrar soporte para subida de múltiples archivos simultáneos.
  - [ ] Añadir botón para exportar el CSV limpio y categorizado.
  - [ ] Implementar persistencia de datos (SQLite) opcional.

## 🤝 Contribuciones

¿Tienes una idea para mejorar los prompts de la IA o una nueva métrica? ¡Los PR son bienvenidos\!

-----

*Hecho con ❤️, Python y mucho café por Matías.*
