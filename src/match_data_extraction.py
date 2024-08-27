# pylint: disable=W0613
# pylint: disable=W0107
# pylint: disable=C0103
# pylint: disable=C0301

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

    return lineup["player_id"].dropna().unique().tolist()


def get_passes(lineup: EventosLineup, jugador: Jugador, jugador2: Jugador) -> int:
    """ Obtiene la cantidad de pases de un jugador a otro en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador que realiza el pase
        jugador2 (Jugador): Jugador que recibe el pase

    Returns:
        int: Cantidad de pases
    """
    passes = lineup[(lineup["player_id"] == jugador) & (lineup["player_id"].shift(-1) == jugador2) & (lineup["type"] == 1) & (lineup["outcome"] == 1)]
    return len(passes)


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


def get_shots(lineup: EventosLineup, jugador: Jugador, OffTarget: bool) -> int:
    """ Obtiene la cantidad de tiros de un jugador en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador que realiza el tiro
        OffTarget (bool): Si se incluyen tiros fuera de puerta

    Returns:
        int: Cantidad de tiros
    """
    if OffTarget:
        shots = lineup[(lineup["player_id"] == jugador) & (lineup["type"].isin[13, 14, 15, 16])]  # 13 and 14 are off target, 15 and 16 are on target
    else:
        shots = lineup[(lineup["player_id"] == jugador) & (lineup["type"].isin[15, 16])]
    return len(shots)


def get_goals(lineup: EventosLineup, jugador: Jugador) -> int:
    """ Obtiene la cantidad de goles de un jugador en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador que anota un gol

    Returns:
        int: Cantidad de goles
    """

    goals = lineup[(lineup["player_id"] == jugador) & (lineup["type"] == 16)]
    return len(goals)


def get_time_played(lineup: EventosLineup, jugador: Jugador) -> int:
    """ Obtiene el tiempo jugado por un jugador en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador

    Returns:
        int: Tiempo jugado en minutos
    """

    if jugador not in get_jugadores(lineup):
        return 0

    return get_lineup_duration(lineup)


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
    if jugador not in get_jugadores(lineup) or jugador2 not in get_jugadores(lineup):
        return 0

    return get_lineup_duration(lineup)


def get_lineup_duration(lineup: EventosLineup) -> int:
    """ Obtiene la duración de un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup

    Returns:
        int: Duración en minutos
    """

    starting_id = lineup["min"].idxmin()
    ending_id = lineup["min"].idxmax()

    starting_min = float(lineup.loc[starting_id, "min"])
    starting_sec = float(lineup.loc[starting_id, "sec"])

    ending_min = float(lineup.loc[ending_id, "min"])
    ending_sec = float(lineup.loc[ending_id, "sec"])

    elapsed_mins = (ending_min + ending_sec / 60) - (starting_min + starting_sec / 60)

    return elapsed_mins
