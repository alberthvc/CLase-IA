# Proyecto: Naive Bayes con Estimación KDE para Mantenimiento Predictivo

**Curso:** IA / Mantenimiento Predictivo  
**Dataset:** `ai4i2020.csv` (AI4I 2020 Predictive Maintenance Dataset)  
**Objetivo:** Predecir la variable **`Machine failure`** (0/1) comparando dos enfoques:
1) **Gaussian Naive Bayes (GNB)** – suposición paramétrica (distribución normal).  
2) **KDE Naive Bayes (KDE‑NB)** – estimación **no paramétrica** de densidades con Kernel Density Estimation (KDE) + ajuste de umbral.

---

## 1) Estructura del proyecto

```
PROYECTO 1/
├─ ai4i2020.csv
├─ proyecto_1.ipynb              # Notebook principal
├─ requirements.txt              # Dependencias del proyecto
├─ .vscode/
│  └─ settings.json              # (Opcional) VS Code apuntando a .venv
├─ figs/                         # Gráficos exportados (al ejecutar el notebook)
├─ cm_gnb_05.csv                 # Matriz de confusión GNB (umbral 0.5)
├─ cm_kde_05.csv                 # Matriz de confusión KDE-NB (umbral 0.5)
├─ cm_kde_opt.csv                # Matriz de confusión KDE-NB (umbral optimizado)
└─ comparativa_nb.csv            # Tabla comparativa de métricas
```

> Si alguna carpeta no existe (p. ej. `figs/`), se crea automáticamente cuando ejecutes las celdas de guardado.

---

## 2) Configuración rápida (Windows + VS Code)

> Requiere **Python 3.10–3.12** (recomendado). En VS Code, abre **PROYECTO 1** como carpeta de trabajo.

### A. Crear/activar entorno virtual
En la **Terminal de VS Code (PowerShell)**, dentro de `PROYECTO 1/`:

```powershell
# 1) Crear venv (si no existe)
python -m venv .venv

# 2) Activar venv
.\.venv\Scripts\Activate.ps1
```

> Si PowerShell bloquea scripts, ejecuta una sola vez (con permisos):  
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### B. Instalar dependencias
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> Si no tienes `requirements.txt`, créalo con esta base:
> ```text
> numpy==2.3.3
> pandas==2.3.3
> scikit-learn==1.4.2
> scipy==1.13.1
> matplotlib==3.9.0
> seaborn==0.13.2
> jupyter==1.0.0
> ipykernel==6.29.5
> notebook==7.2.2
> ```

---

## 3) Columnas usadas

**Target (y):** `Machine failure` (0 = no falla, 1 = falla).

**Features numéricas:**
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

**Feature categórica:**  
- `Type` → se codifica con One-Hot Encoding (`Type_L`, `Type_M`, `Type_H`).

**No usar como features (ID/fuga de información):** `UDI`, `Product ID` y las etiquetas de fallas específicas `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (derivadas del target).

---

## 4) Flujo del notebook `proyecto_1.ipynb`

1. **Carga & limpieza mínima** del dataset (`ai4i2020.csv`), revisión de dtypes y valores.
2. **Construcción de `X` y `y`**  
   - Selección de columnas numéricas + One‑Hot para `Type`.
3. **Split estratificado 80/20** con `train_test_split` (mantiene proporción de la clase minoritaria).
4. **Baseline – Gaussian Naive Bayes (GNB)**  
   - Entrenamiento, **accuracy**, `classification_report`, **matriz de confusión**.
   - Guardado: `cm_gnb_05.csv`.
5. **KDE Naive Bayes (KDE‑NB)**  
   - Estimación de densidades por clase con KDE.  
   - Predicción probabilística y evaluación con **umbral 0.5**.  
   - Guardado: `cm_kde_05.csv`.
6. **Ajuste de umbral (optimización por F1 positiva)**  
   - Búsqueda de umbral en validación para **maximizar F1 de la clase 1** (fallas).  
   - Re‑evaluación en `X_test` con ese umbral óptimo.  
   - Guardado: `cm_kde_opt.csv`.
7. **Comparativa de modelos**  
   - Consolidación de métricas (**accuracy, precision/recall/F1**) en `comparativa_nb.csv`.
8. **Gráficos & reportes**  
   - Curva **Precision‑Recall** (enfocada en clases desbalanceadas),  
   - Curva **ROC**,  
   - **Matriz de confusión** anotada,  
   - Distribuciones de variables y **calor de correlación**.  
   - Guardado en `figs/`.

---

## 5) Cómo ejecutar y reproducir

```powershell
# En la terminal con el venv activo:
code .                 # (opcional) abrir VS Code en la carpeta
jupyter notebook       # o "jupyter lab" si prefieres
```

- Abre `proyecto_1.ipynb` y ejecuta las celdas **de arriba hacia abajo**.  
- Al finalizar, encontrarás:
  - CSVs: `cm_gnb_05.csv`, `cm_kde_05.csv`, `cm_kde_opt.csv`, `comparativa_nb.csv`  
  - Gráficos en `figs/`.

> **Nota:** Los valores exactos pueden variar ligeramente por el `random_state` y por detalles de validación. El foco es comparar la **mejora de F1/recall** de la clase positiva con KDE‑NB y/o con **ajuste de umbral** frente a GNB.

---

## 6) Resultados esperados (guía rápida)

- **GNB (umbral 0.5):** baseline con accuracy alto pero **F1 de la clase 1** relativamente bajo por el desbalance.  
- **KDE‑NB (umbral 0.5):** suele **mejorar recall** (y por tanto F1) de la clase 1 frente a GNB.  
- **KDE‑NB (umbral óptimo):** equilibrio mejorado **precision–recall** para fallas, sacrificando poca accuracy total.

Revisa `comparativa_nb.csv` y las matrices `cm_*.csv` para ver los números concretos de tu corrida.

---

## 7) Buenas prácticas y siguientes pasos

- Reportar **accuracy + precision/recall/F1** (macro/weighted y por clase).  
- Priorizar **Precision‑Recall** y **F1 de la clase 1** en conjuntos desbalanceados.  
- Validar con **K‑Fold estratificado** o **TimeSeriesSplit** (si hay orden temporal).  
- Probar **otros kernels/bandwidths** en KDE y **calibración** (`CalibratedClassifierCV`).  
- Comparar con **Logistic Regression**, **RandomForest**, **XGBoost/LightGBM**.  
- Ingeniería de variables (p. ej., interacciones, polinomios, *standard scaling*).

---

## 8) Solución de problemas

- **No activa el venv / error de scripts en PowerShell:**  
  Ejecuta una sola vez en la sesión:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```

- **Import “could not be resolved from source” en VS Code:**  
  1) Verifica que el kernel/interprete es `.venv\Scripts\python.exe`.  
  2) `pip install -r requirements.txt`.  
  3) Reinicia VS Code si persiste.

- **Faltan carpetas (`figs/`):** se crean cuando corres las celdas que guardan gráficos.

---

## 9) Créditos y licencia

- **Dataset:** *AI4I 2020 Predictive Maintenance Dataset* (UCI Machine Learning Repository).  
- **Código:** Uso académico. Ajusta y reutiliza con referencia a este proyecto.

---

## 10) Contacto

Si necesitas ayuda para replicar los experimentos, agregar más modelos o preparar la presentación (PPT), abre un issue o escribe un comentario en el notebook.
