""" Procesamiento de eventos de partidos de futbol del Excel """

from typing import List, Tuple

import pandas as pd
from src.excel_xml_util import change_in_zip
from src.futbol_types import Temporada, Partido, EventosLineup


def leer_excel(path: str) -> Temporada:
    """Lee un archivo Excel con eventos de partidos de futbol

    Args:
        path (str): Ruta al archivo Excel

    Returns:
        Temporada: Eventos de todos los partidos de una temporada
        Columns: ['season_id', 'match_id', 'home_team_id', 'home_team_name',
                  'away_team_id', 'away_team_name', 'id', 'event_id', 'date', 'time',
                  'period_id', 'min', 'sec', 'team_id', 'player_id', 'playerName',
                  'playerPosition', 'x', 'y', 'type', 'description', 'outcome']
    """

    # Fix for the "synchVertical" property in the Excel files
    # Pandas does not support reading this property
    change_in_zip(
        path,
        # the problematic property is found in the worksheet xml files
        name_filter="xl/worksheets/*.xml",
        change=lambda d: d.replace(b' synchVertical="1"', b" "),
    )

    return pd.read_excel(path)


def separar_partidos(df: Temporada) -> List[Partido]:
    """Separa los eventos de una temporada en partidos individuales

    Args:
        df (Temporada): Eventos de todos los partidos de una temporada

    Returns:
        List[Partido]: Eventos de cada partido
    """

    # With groupby
    # return [partido for _, partido in df.groupby('match_id')]

    return [df[df["match_id"] == match_id] for match_id in df["match_id"].unique()]


def separar_partido_en_equipo_pov(df: Partido) -> Tuple[Partido, Partido]:
    """Separa los eventos de un partido en dos DataFrames, uno por equipo

    Args:
        df (Partido): Eventos de un partido

    Returns:
        Tuple[Partido, Partido]: Eventos de cada equipo
    """

    return [
        df[df["team_id"] == team_id] for team_id in df["team_id"].unique()
    ]


def separar_partido_del_equipo_en_lineups(df: Partido) -> List[EventosLineup]:
    """Separa los eventos de un equipo en dos DataFrames, uno por lineup

    Args:
        df (Partido): Eventos de un equipo en un partido

    Returns:
        List[EventosLineup]: Eventos de cada lineup
    """

    # Reset index
    df = df.reset_index(drop=True)

    # Filtramos los eventos de cambios de alineación
    changes_events_idx = (df[(df["type"] == 18) | (df["type"] == 19)]).index
    print(changes_events_idx)
    
    # Crear una lista para almacenar los dataframes resultantes
    lineups_events = []

    # Recorrer los índices de los eventos separadores y crear los nuevos dataframes
    start = 0
    for i in range(0, len(changes_events_idx), 2):
        end = changes_events_idx[i]  
        print(start, end)
        lineups_events.append(df.iloc[start:end])
        start = end + 2

    # Añadir el último lineup
    if start < len(df):
        lineups_events.append(df.iloc[start:])

    # Si no hay cambios de alineación, añadir el dataframe completo
    if len(lineups_events) == 0:
        lineups_events.append(df)

    return lineups_events
