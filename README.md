# Titanic - Análisis y Preprocesamiento de Datos

Análisis exploratorio y preprocesamiento de datos sobre el dataset
[Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)
de Kaggle. Proyecto individual para la materia de Big Data (práctica de
Git, GitHub y reproducibilidad de proyectos de datos).

## Dataset

891 pasajeros del Titanic (`train.csv`) con las columnas:

- `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`
- `SibSp` (hermanos/cónyuges a bordo), `Parch` (padres/hijos a bordo)
- `Ticket`, `Fare`, `Cabin`, `Embarked`

El archivo crudo se incluye en `data/raw/train.csv` para que el proyecto
sea reproducible sin necesidad de credenciales de Kaggle. El script
`src/download_data.py` documenta el origen del dato y cómo volver a
descargarlo si se desea.

## Estructura del proyecto

```
titanic-analysis/
├── data/
│   ├── raw/            # dataset original (train.csv)
│   └── processed/      # datos limpios generados por el pipeline
├── notebooks/          # exploración auxiliar (opcional)
├── src/
│   ├── download_data.py
│   ├── preprocess.py
│   └── eda.py
├── reports/
│   └── figures/        # gráficas generadas por el EDA
├── requirements.txt
└── README.md
```

## Cómo reproducir el proyecto

1. Clonar el repositorio:

   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd titanic-analysis
   ```

2. Crear y activar un entorno virtual:

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

4. El dataset crudo ya está incluido en `data/raw/train.csv`. Si se
   prefiere descargarlo directamente desde Kaggle, ver las instrucciones
   en `src/download_data.py`.

5. Ejecutar el pipeline:

   ```bash
   python src/preprocess.py
   python src/eda.py
   ```

   Esto genera `data/processed/titanic_clean.csv` y las gráficas en
   `reports/figures/`.

## Autor

Elías F. Maltrana — proyecto individual, materia de Big Data.
