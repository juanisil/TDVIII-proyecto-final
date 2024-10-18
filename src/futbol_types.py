# pylint: disable=W0613
# pylint: disable=C0103
# pylint: disable=C0301
# pylint: disable=E0401

""" Tipos de datos usados en el análisis de datos de fútbol """

from typing import NewType

import pandas as pd
import numpy as np

# Definir tipos específicos
Evento = NewType('Evento', pd.Series)
Partido = NewType('Partido', pd.DataFrame)
Temporada = NewType('Temporada', pd.DataFrame)  # Eventos de todos los partidos de una temporada
EventosLineup = NewType('EventosLineup', pd.DataFrame)  # Eventos de un lineup en un partido
Jugador = NewType('Jugador', str)
TransitionMatrix = NewType('TransitionMatrix', np.ndarray)  # Con forma (14, 14), la suma de cada fila es 1
