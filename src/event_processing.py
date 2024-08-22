""" Procesamiento de eventos de partidos de futbol del Excel """

from typing import List, Tuple
from futbol_types import Temporada, Partido, EventosLineup


def leer_excel(path: str) -> Temporada:
    """Lee un archivo Excel con eventos de partidos de futbol

    Args:
        path (str): Ruta al archivo Excel

    Returns:
        Temporada: Eventos de todos los partidos de una temporada
    """
    pass


def separar_partidos(df: Temporada) -> List[Partido]:
    """Separa los eventos de una temporada en partidos individuales

    Args:
        df (Temporada): Eventos de todos los partidos de una temporada

    Returns:
        List[Partido]: Eventos de cada partido
    """

    pass


def separar_partido_en_equipo(df: Partido) -> Tuple[Partido, Partido]:
    """ Separa los eventos de un partido en dos DataFrames, uno por equipo

    Args:
        df (Partido): Eventos de un partido

    Returns:
        Tuple[Partido, Partido]: Eventos de cada equipo
    """

    pass


def separar_partido_del_equipo_en_lineups(df: Partido) -> List[EventosLineup]:
    """ Separa los eventos de un equipo en dos DataFrames, uno por lineup

    Args:
        df (Partido): Eventos de un equipo en un partido

    Returns:
        List[EventosLineup]: Eventos de cada lineup
    """
    pass
