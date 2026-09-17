# Titanic - Análisis y Preprocesamiento de Datos

Proyecto individual de la actividad **Git, GitHub y reproducibilidad de
proyectos de datos** (Big Data). Consiste en limpiar, preprocesar y
analizar exploratoriamente el dataset Titanic para identificar qué
características de los pasajeros se asociaron con la supervivencia.

## Dataset

- **Nombre:** Titanic - Machine Learning from Disaster
- **Fuente:** [https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)
- **Descripción:** 891 pasajeros del Titanic (`train.csv`), con columnas
  `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`
  (hermanos/cónyuges a bordo), `Parch` (padres/hijos a bordo), `Ticket`,
  `Fare`, `Cabin` y `Embarked`.

El archivo crudo se incluye en `data/raw/train.csv` para que el proyecto
sea reproducible sin necesidad de credenciales de Kaggle. El script
`src/download_data.py` documenta el origen del dato y cómo volver a
descargarlo desde la API oficial si se desea.

## Objetivo

Analizar la información disponible de los pasajeros del Titanic para
identificar qué características (sexo, clase, edad, viajar solo o
acompañado) están asociadas con la probabilidad de supervivencia.
No se entrena ningún modelo de Machine Learning: el alcance de esta
práctica es limpieza, preprocesamiento, análisis exploratorio y
visualización.

## Requisitos

- Python 3.11+ (ver nota de compatibilidad más abajo)
- Las dependencias listadas en `requirements.txt`

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
├── .gitignore
└── README.md
```

## Instalación

1. Clonar el repositorio:

   ```bash
   git clone <URL_DEL_REPOSITORIO>
   ```

2. Entrar al proyecto:

   ```bash
   cd titanic-analysis
   ```

3. Crear el entorno virtual:

   ```bash
   python3 -m venv venv
   ```

4. Activarlo e instalar las dependencias:

   ```bash
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Ejecución

El dataset crudo ya está incluido en `data/raw/train.csv`, así que no es
necesario descargar nada adicional. Ejecutar el pipeline en este orden:

```bash
python src/preprocess.py
python src/eda.py
```

- `preprocess.py` limpia el dataset y genera `data/processed/titanic_clean.csv`.
- `eda.py` corre el análisis exploratorio, imprime las respuestas a las
  preguntas de la sección siguiente y guarda las gráficas en
  `reports/figures/`.

**Nota de compatibilidad:** `requirements.txt` no fija versiones exactas
para evitar problemas de ruedas (wheels) no disponibles en versiones muy
recientes de Python (ver comentario en la sección 5 del documento de
evidencia de entrega, donde esto causó un fallo al reproducir el proyecto
de mi compañero). Si tu `pip install` falla al compilar `pandas`, usa
Python 3.11–3.12.

## Tratamiento de valores faltantes

El dataset original tiene valores faltantes en tres columnas:

| Columna    | Faltantes | % del total | Tratamiento |
|------------|-----------|-------------|-------------|
| `Age`      | 177       | 19.9%       | Se imputó con la mediana de edad de su grupo (`Pclass`, `Sex`), en vez de una mediana global, porque la edad varía notablemente entre clases y sexos en este dataset. |
| `Cabin`    | 687       | 77.1%       | Se eliminó la columna por ser demasiado dispersa para imputar de forma confiable, pero se conservó como bandera binaria `has_cabin` (tener cabina registrada correlaciona con clase/tarifa). |
| `Embarked` | 2         | 0.2%        | Se imputó con la moda (puerto más frecuente), al ser solo 2 registros. |

No se encontraron registros duplicados en el dataset (verificado con
`df.duplicated().sum()` en `preprocess.py`).

## Variables nuevas creadas

- `has_cabin`: 1 si el pasajero tiene cabina registrada, 0 si no.
- `family_size`: `SibSp + Parch + 1` (tamaño total de la familia a bordo).
- `is_alone`: 1 si el pasajero viajaba solo (`family_size == 1`).
- `title`: título extraído del nombre (Mr, Mrs, Miss, Master, Rare...).
- `age_group`: categoría de edad (`Child` ≤12, `Teen` 13-18, `Adult` 19-60,
  `Senior` >60).

## Análisis realizados

**1. ¿Qué porcentaje de pasajeros sobrevivió?**
38.4% de los pasajeros sobrevivió (342 de 891).

**2. ¿Cómo cambia la supervivencia entre hombres y mujeres?**
El sexo es el factor más determinante: 74.2% de las mujeres sobrevivió,
frente a solo 18.9% de los hombres.

**3. ¿Cómo cambia la supervivencia según la clase del pasajero?**
1ª clase: 63.0% — 2ª clase: 47.3% — 3ª clase: 24.2%. A mayor clase
(mejor categoría), mayor tasa de supervivencia.

**4. ¿Qué grupos de edad presentan mayor supervivencia?**
Child: 58.0% — Teen: 42.9% — Adult: 36.6% — Senior: 22.7%. Los niños
tuvieron la mayor tasa de supervivencia; la protección "mujeres y niños
primero" se refleja claramente en los datos.

**5. ¿Viajar solo o acompañado parece estar relacionado con la
supervivencia?**
Viajar acompañado (family_size 2-4) se asoció con mayor supervivencia
(50.6%) que viajar solo (30.4%). Familias muy grandes (5+) volvieron a
tener tasas bajas, probablemente por dificultad para evacuar juntos.

Estas 5 respuestas (más de las 4 mínimas requeridas) se calculan e
imprimen automáticamente al ejecutar `python src/eda.py`.

## Visualizaciones

Generadas en `reports/figures/` (5 gráficas, más de las 3 mínimas
requeridas):

- `survival_by_sex.png` — tasa de supervivencia por sexo.
- `survival_by_class.png` — tasa de supervivencia por clase de pasajero.
- `age_distribution.png` — distribución de edad por supervivencia.
- `survival_by_family_size.png` — tasa de supervivencia por tamaño de familia.
- `survival_by_age_group.png` — tasa de supervivencia por grupo de edad.

## Resultados y conclusiones

- El **sexo** fue el factor más determinante de supervivencia
  (74.2% mujeres vs. 18.9% hombres), consistente con el protocolo de
  evacuación "mujeres y niños primero".
- La **clase del pasajero** también importó mucho: viajar en 1ª clase
  casi triplicó la probabilidad de supervivencia frente a 3ª clase,
  probablemente por la cercanía de los camarotes de 1ª clase a los
  botes salvavidas y por prioridad de acceso.
- Los **niños** tuvieron la mayor tasa de supervivencia por grupo de
  edad, y los adultos mayores la menor.
- **Viajar acompañado** (en familias pequeñas de 2 a 4 personas) se
  asoció con mejor supervivencia que viajar solo o en familias muy
  grandes.
- En conjunto, los datos muestran que la supervivencia en el Titanic no
  fue aleatoria: estuvo fuertemente influida por el sexo, la clase
  social y la composición familiar del pasajero.

## Autor

Elías F. Maltrana — proyecto individual, materia de Big Data.
