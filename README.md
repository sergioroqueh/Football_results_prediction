# Predicción de Resultados de Fútbol

<p align="center">
  <img src="docs/Flujo de trabajo - Proyecto.jpg" alt="Diagrama del flujo de trabajo del proyecto" width="950">
</p>

## Descripción

Este proyecto desarrolla un pipeline completo para predecir el resultado de partidos de fútbol a partir de datos históricos de temporadas anteriores.

El flujo incluye:
- Ingesta de datos por temporada.
- Limpieza y unificación de formatos.
- Feature engineering temporal y global.
- Entrenamiento de modelos predictivos.
- Generación de probabilidades para tomar decisiones.

La idea principal es construir un sistema que respete el orden temporal de los datos y evite fugas de información.

## Stack

- Python
- pandas
- numpy
- scikit-learn
- matplotlib / seaborn
- Jupyter Notebook

## Flujo de trabajo

1. **Ingesta de datos**
   - Carga de CSV por temporada.
   - Lectura de archivos con `pandas`.
   - Unificación de columnas y formatos.

2. **Limpieza**
   - Eliminación de columnas basura.
   - Eliminación de duplicados.
   - Eliminación de filas vacías.
   - Conversión de tipos.
   - Ordenación por fecha.

3. **Feature engineering**
   - Construcción de variables por partido.
   - Cálculo de forma reciente.
   - Variables de goles a favor y en contra.
   - Contexto local/visitante.
   - Variables globales tras unir temporadas.
   - Histórico H2H.

4. **Modelado**
   - Definición de la variable objetivo.
   - Selección de features.
   - Separación temporal train/test.
   - Entrenamiento del modelo.
   - Evaluación con métricas.

5. **Predicción**
   - Construcción del estado actual de los equipos.
   - Generación de features en tiempo real.
   - Cálculo de probabilidad de victoria.
   - Decisión final: apostar o no apostar.

## Resultados

El objetivo del modelo no es solo predecir un ganador, sino estimar probabilidades útiles para tomar decisiones más informadas.

Las métricas y resultados finales pueden incluir:
- Accuracy.
- Log loss.
- Matriz de confusión.
- Comparación entre temporadas.
- Rendimiento sobre partidos no vistos.

## Estructura del proyecto

```bash
.
├── README.md
├── docs/
│   └── Flujo-de-Trabajo.jpg
├── data/
├── notebooks/
├── src/
└── models/
```

## Cómo usarlo

1. Clona el repositorio.
2. Instala dependencias.
3. Ejecuta el notebook principal o el script de entrenamiento.
4. Revisa las predicciones generadas.

## Próximos pasos

- Mejorar la calibración de probabilidades.
- Añadir más variables contextuales.
- Comparar varios modelos.
- Evaluar ROI simulado con cuotas reales.
- Automatizar la actualización de datos por temporada.

## Autor

Proyecto desarrollado por Sergio Roque Hernández.