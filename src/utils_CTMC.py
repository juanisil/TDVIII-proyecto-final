# pylint: disable=W0613
# pylint: disable=C0103
# pylint: disable=C0301
# pylint: disable=E0401

"""
    Contiene las funciones necesarias para calcular el modelo de transición de estados de un equipo en un partido de futbol
"""

import numpy as np
from src.event_processing import separar_partido_del_equipo_en_lineups
from src.futbol_types import EventosLineup, Jugador, Partido, TransitionMatrix
from src.match_data_extraction import get_gains, get_jugadores, get_lineup_duration, get_losses, get_passes, get_shots

# transition r_g(G,p_i): from the gain state to a player p_i as
# r_g(G,p_i)= g_g(p_i)/t_g(p_i, p_i)                                                        (4)

# the rate of passes between two players
# r_g(p_i,p_j) = m_g(p_i,p_j)/t_g(p_i,p_j)                                                  (5)

# the rates from a player to losses
# r_g(p_i, L) = l_g(p_i)/t_g(p_i,p_i)                                                       (6)

# the rates from a player to shots
# r_g(p_i, S) = s_g(p_i)/t_g(p_i,p_i)                                                       (7)


def get_ratio_gains(lineup: EventosLineup, jugador: Jugador) -> float:
    """Calcula la tasa de posesiones ganadas por un jugador en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador que gana posesiones

    Returns:
        r G p1 (float): Tasa de posesiones ganadas
    """
    gains = get_gains(lineup, jugador)
    time = get_lineup_duration(lineup)
    if time == 0:
        return 0

    return gains / time


def get_ratio_loss(lineup: EventosLineup, jugador: Jugador) -> float:
    """Calcula la tasa de posesiones perdidas por un jugador en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador que pierde posesiones

    Returns:
        r p1 L (float): Tasa de posesiones perdidas
    """

    losses = get_losses(lineup, jugador)
    time = get_lineup_duration(lineup)

    if time == 0:
        return 0

    return losses / time


def get_ratio_shots(lineup: EventosLineup, jugador: Jugador) -> float:
    """Calcula la tasa de tiros de un jugador en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador que realiza tiros

    Returns:
        r pi S (float): Tasa de tiros
    """

    shots = get_shots(lineup, jugador)
    time = get_lineup_duration(lineup)

    if time == 0:
        return 0

    return shots / time


def get_ratio_passes(
    lineup: EventosLineup, jugador: Jugador, jugador2: Jugador
) -> float:
    """Calcula la tasa de pases de un jugador a otro en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador que realiza el pase
        jugador2 (Jugador): Jugador que recibe el pase

    Returns:
        r pi, pj (float): Tasa de pases
    """

    passes = get_passes(lineup, jugador, jugador2)
    time = get_lineup_duration(lineup)

    if time == 0:
        return 0

    return passes / time


def build_R(lineup: EventosLineup) -> TransitionMatrix:
    """Construye la matriz de transición de estados R (Como Q pero sin Normalización) de un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup

    Returns:
        R (TransitionMatrix): Matriz de transición de estados sin normalizar
    """

    R: TransitionMatrix = np.zeros((14, 14))

    # Loss to Loss
    R[12, 12] = 1

    # Shots to Shot
    R[13, 13] = 1

    # Col 1 is Every state to P1
    # ...
    # Col 11 is Every state to P11
    # Col 12 is Every state to Loss
    # Col 13 is Every state to Shot

    # Players should be each player in the lineup
    players = get_jugadores(lineup)
    for i, player in enumerate(players):

        R[0, i + 1] = get_ratio_gains(lineup, player)

        for j, player2 in enumerate(players):

            R[i + 1, j + 1] = get_ratio_passes(lineup, player, player2)

        R[i + 1, 12] = get_ratio_loss(lineup, player)
        R[i + 1, 13] = get_ratio_shots(lineup, player)

    return R


def build_Q(R: TransitionMatrix) -> TransitionMatrix:
    """ Dada una matriz de transición de estados R,
        construye la matriz de transición de estados Q normalizando R

    Args:
        R (TransitionMatrix): Matriz de transición de estados sin normalizar

    Returns:
        Q (TransitionMatrix): Matriz de transición de estados normalizada
    """

    # Normaliza R

    # the transition probability q_g(U, V) between any two states U and V is given by

    #                                    r_g(U,V)
    # q_g(U, V) =  -----------------------------------------------------------                  (8)
    #              r_g(U,G) + r_g(U,S) + r_g(U,L) + sum_{i=1}^{11}{r_g(U,p_i)}

    Q: TransitionMatrix = np.array(R.copy())

    for i in range(14):
        if np.sum(Q[i, :]) != 0:
            # Normalize each row
            Q[i, :] = Q[i, :] / np.sum(Q[i, :])

    return Q


def psl_estimator(Q: TransitionMatrix) -> float:
    """ Calcula la probabilidad de perder la pelota
        antes de tirar al arco de un lineup
        a partir de la matriz de transición de estados Q

    Args:
        Q (TransitionMatrix): Matriz de transición de estados normalizada

    Returns:
        PSL (float): Probabilidad de perder la pelota antes de tirar al arco
    """

    # Define the following block decomposition of Q_g(A)

    #           |  T_{12x12}   R_{12x2}  |
    # Q_g(A) =  |
    #           |  0_{2x12}    I_{2x2}   |

    # where T contains the transition probabilities between transient states,
    # R contains the transition probabilities from transient to absorbing states,
    # 0 is a block of all zeros and I is an identity block.
    #
    # Leveraging this decomposition, the probability of shot before loss p_g(A) can be estimated from Qq(A) as we state in the following definition.

    # p^_g(A) = [1, 0_{1×11}](I_{12x12} - T)^{-1} R[0, 1]^T

    # [1, 0_{1×11}] is a row vector of length 12 with a 1 in the first position and zeros elsewhere.
    # [0, 1]^T is a column vector of length 2 with a 1 in the second position and zeros elsewhere.

    # The matrix (I_{12x12} - T)^{-1} is the inverse of the matrix I_{12x12} - T.
    # The inverse of a matrix is a matrix that when multiplied by the original matrix gives an identity matrix.

    T: np.ndarray = Q[:12, :12]
    R: np.ndarray = Q[:12, 12:]

    M = np.eye(12) - T

    psl = 0

    try:
        M_inv = np.linalg.inv(M)
        psl = np.dot(np.dot(np.array([1] + [0] * 11), M_inv), R).dot(np.array([0, 1]).T)
    except np.linalg.LinAlgError as e:
        print("Error: La matriz es singular y no se puede invertir.")
        print(e)

    return psl


def team_psl(equipo: Partido) -> float:
    """
        Calcula la probabilidad de perder la pelota antes de tirar al arco de un equipo en un partido
        A partir del psl de cada lineup del equipo
        Devuelve el promedio de psl ponderado por la duración de cada lineup

    Args:
        equipo (Partido): Eventos de un equipo en un partido

    Returns:
        float: Probabilidad de perder la pelota antes de tirar al arco
    """

    lineups = separar_partido_del_equipo_en_lineups(equipo)
    psls = np.array([psl_estimator(build_Q(build_R(lineup))) for lineup in lineups])
    lineup_durations = np.array([get_lineup_duration(lineup) for lineup in lineups])

    return np.average(psls, weights=lineup_durations)
