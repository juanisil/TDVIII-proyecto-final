""" Módulo para extraer datos de los partidos """

from typing import List
from src.futbol_types import EventosLineup, Jugador


def get_jugadores(lineup: EventosLineup) -> List[Jugador]:
    """ Obtiene los jugadores de un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup

    Returns:
        List[Jugador]: Jugadores del lineup
    """

    return []


def get_passes(lineup: EventosLineup, jugador: Jugador, jugador2: Jugador) -> int:
    """ Obtiene la cantidad de pases de un jugador a otro en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador que realiza el pase
        jugador2 (Jugador): Jugador que recibe el pase

    Returns:
        int: Cantidad de pases
    """

    pass


def get_gains(lineup: EventosLineup, jugador: Jugador) -> int:
    """ Obtiene la cantidad de posesiones ganadas por un jugador en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador que gana una posesión

    Returns:
        int: Cantidad de posesiones ganadas
    """

    pass


def get_loss(lineup: EventosLineup, jugador: Jugador) -> int:
    """ Obtiene la cantidad de posesiones perdidas por un jugador en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador que pierde una posesión

    Returns:
        int: Cantidad de posesiones perdidas
    """

    pass


def get_shots(lineup: EventosLineup, jugador: Jugador) -> int:
    """ Obtiene la cantidad de tiros de un jugador en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador que realiza el tiro

    Returns:
        int: Cantidad de tiros
    """

    pass


def get_time_played(lineup: EventosLineup, jugador: Jugador) -> int:
    """ Obtiene el tiempo jugado por un jugador en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador

    Returns:
        int: Tiempo jugado en minutos
    """

    # Propio del lineup
    pass


def get_shared_time(lineup: EventosLineup, jugador: Jugador, jugador2: Jugador) -> int:
    """ Obtiene el tiempo jugado por dos jugadores en un lineup
    
    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador
        jugador2 (Jugador): Jugador

    Returns:
        int: Tiempo jugado en minutos
    """

    # Propio del lineup
    pass


def get_lineup_duration(lineup: EventosLineup) -> int:
    """ Obtiene la duración de un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup

    Returns:
        int: Duración en minutos
    """

    pass
