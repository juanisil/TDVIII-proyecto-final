# Cómo encontrar el mejor jugador para tu Equipo de Fútbol

## Integrantes

-   Tomás Glauberman - 21F78 \| <tglauberman@mail.utdt.edu>
-   Ignacio Pardo - 21R1160 \| <ipardo@mail.utdt.edu>
-   Juan Ignacio Silvestri - 21Q111 \| <jsilvestri@mail.utdt.edu>

## Descripción

Código fuente del proyecto final de la Licenciatura en Tecnología
Digital de la Universidad Torcuato Di Tella.

## Estructura

El proyecto está dividido las siguientes carpetas principales:

-   `src/`: Código “en producción” del proyecto. Incluye un módulo de
    Python con funciones y clases que se utilizan en el análisis de
    datos.
-   `exploratory/`: Notebooks de Jupyter y otros archivos de exploración
    de datos.
-   `tests/`: Archivos de tests unitarios y de integración sobre el
    código en la carpeta `src/`.
-   `SampleData/`: Datos de OPTA de la English Premier League 2012/2013.
-   `gui/`: Interfaz gráfica para la exploración de la investigación.

## Source Code

-   `src/event_processing.py`: Procesamiento de eventos de partidos de
    futbol del Excel

-   `src/excel_xml_util.py`: Modulo para trabajar con archivos XML de
    OPTA en Pandas

-   `src/futbol_types.py`: Definición de tipos de datos para el análisis
    de futbol

-   `src/match_data_extraction.py`: Módulo para extraer eventos y datos
    de los partidos

-   `src/epl_player_data_utils.py`: Datos extra de jugadores de la EPL
    por ID de OPTA.

-   `src/utils_CTMC.py`: Funciones para calcular el modelo de transición
    de estados de un equipo en un partido de futbol.

-   `src/bayesian_PSL.py`: Clases para el análisis bayesiano del PSL.

-   `src/Player2Vec.py`: Player2Vec a partir de Node2Vec sobre Grafo
    Full de Transición de Estados de Jugadores

-   `src/p2v_dist.py`: Player2Vec to Player Stats Distribution Model

-   `src/p2v_dist.py`: Modelo Predictivo de Distribuciones de Ratios de
    Transición

## Algunos Notebooks

-   `exploratory/emb_viz.ipynb`: Visualización de embeddings de
    jugadores con Player2Vec.
-   `exploratory/dataset_exp.ipynb`: Breve exploración del dataset de
    OPTA.
-   `exploratory/p2v_dist_model_oos.ipynb`: Modelo Predictivo de
    Distribuciones de Ratios de Transición, con Validación
    Out-of-Sample. `src/p2v_dist.py`.
-   `exploratory/OPTA_XML_Read.ipynb`: Lectura de datos de OPTA en
    formato XML.
-   `exploratory/player_exp.ipynb`: Exploración de datos de jugadores.
    Cálculo de Opta Points como ejemplo.
-   `exploratory/player_q_dist.ipynb`: Análisis de la distribución de
    los ratios de transición de jugadores.
-   `exploratory/psl_grad.ipynb`: Test de Sensitividad de la función
    PSL.
-   `exploratory/PSL_StastsBomb.ipynb`: Cálculo de PSL con datos de
    StatsBomb.
-   `exploratory/PSL_sample_data.ipynb`: Cálculo de PSL con datos de
    OPTA.
-   `exploratory/case_study.ipynb`: Caso de estudio de jugadores de la
    EPL (OPTA).

## Usage

El proyecto requiere Python 3.10.14 y las dependencias listadas en
`requirements.txt`. En el archivo Makefile se encuentran los comandos
para instalar las dependencias, correr los tests y correr la interfaz
gráfica de exploración de la investigación.

Para instalar las dependencias necesarias, ejecutar el siguiente
comando:

``` bash
make install
```

Alternativamente se puede crear un entorno de Conda con las dependencias
necesarias ejecutando el siguiente comando:

``` bash
make create_conda_env
```

Para correr los tests unitarios y de integración, ejecutar el siguiente
comando:

``` bash
make test
```

Para correr la interfaz gráfica de exploración de la investigación,
ejecutar el siguiente comando:

``` bash
make viz
```

## Referencias

Bawa, V. S. (1982). Stochastic dominance: A research bibliography.
*Management Science*, *28*, 698–712.
<https://doi.org/10.1287/mnsc.28.6.698>

Bergstra, J., Komer, B., Eliasmith, C., Yamins, D., & Cox, D. D. (2015).
Hyperopt: A python library for model selection and hyperparameter
optimization. *Computational Science & Discovery*, *8*, 014008.
<https://doi.org/10.1088/1749-4699/8/1/014008>

Brunetti, D., Ceria, S., Durán, G., Durán, M., Farall, A., Marucho, N.,
& Mislej, P. (2024). *Data science models for football scouting: The
racing de santander case study*. 33rd European Conference on Operational
Research.
<https://ic.fcen.uba.ar/uploads/files/Euro%202024%20-%20Data%20Science%20models%20for%20Football%20Scouting%20The%20Racing%20de%20Santander%20case%20study%20-%20REVISED.pdf>

Green, S. (2012). *Assessing the performance of premier league
goalscorers*. Stats Perform.
<https://www.statsperform.com/resource/assessing-the-performance-of-premier-league-goalscorers/>

Grover, A., & Leskovec, J. (2016). *node2vec: Scalable feature learning
for networks*. arXiv.org. <https://arxiv.org/abs/1607.00653>

Huang, E., Segarra, S., Gallino, S., & Ribeiro, A. (n.d.). *How to find
the right player for your soccer team?*

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient
estimation of word representations in vector space*. arXiv.org.
<https://arxiv.org/abs/1301.3781>

*Opta data from stats perform*. (n.d.). Stats Perform.
<https://www.statsperform.com/opta/>

PyTorch Forums, A. N. K. -. (2022). *Jensen shannon divergence*. PyTorch
Forums.
<https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/13>

Rahimian, P., Van Haaren, J., & Toka, L. (2023). Towards maximizing
expected possession outcome in soccer. *International Journal of Sports
Science & Coaching*, 174795412311544.
<https://doi.org/10.1177/17479541231154494>

Tippett, J. (2019). *The expected goals philosophy: A game-changing way
of analysing football*. Independently Published.

Vulcano, G. (n.d.). *Decision under risk - module IV - NYU stern -
master of science in business analytics*.
