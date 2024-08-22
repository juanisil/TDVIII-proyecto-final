""" Contiene las funciones necesarias para calcular el modelo de transición de estados de un equipo en un partido de futbol """

from futbol_types import EventosLineup, Jugador, TransitionMatrix

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

    pass


def get_ratio_loss(lineup: EventosLineup, jugador: Jugador) -> float:
    """Calcula la tasa de posesiones perdidas por un jugador en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador que pierde posesiones

    Returns:
        r p1 L (float): Tasa de posesiones perdidas
    """

    pass


def get_ratio_shots(lineup: EventosLineup, jugador: Jugador) -> float:
    """Calcula la tasa de tiros de un jugador en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador que realiza tiros

    Returns:
        r pi S (float): Tasa de tiros
    """

    pass


def get_ratio_passes(
    lineup: EventosLineup, jugador: Jugador, jugador2: Jugador
) -> float:
    """ Calcula la tasa de pases de un jugador a otro en un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
        jugador (Jugador): Jugador que realiza el pase
        jugador2 (Jugador): Jugador que recibe el pase

    Returns:
        r pi, pj (float): Tasa de pases
    """

    pass


def build_R(lineup: EventosLineup) -> TransitionMatrix:
    """ Construye la matriz de transición de estados R (Como Q pero sin Normalización) de un lineup

    Args:
        lineup (EventosLineup): Eventos de un lineup
    
    Returns:
        R (TransitionMatrix): Matriz de transición de estados sin normalizar
    """

    pass


def build_Q(R: TransitionMatrix) -> TransitionMatrix:
    """ Dada una matriz de transición de estados R, construye la matriz de transición de estados Q normalizando R

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

    pass


def psl_estimator(Q: TransitionMatrix) -> float:
    """ Calcula la probabilidad de perder la pelota antes de tirar al arco de un lineup a partir de la matriz de transición de estados Q

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

    pass
