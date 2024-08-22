Separar el excel en partidos
Separar los partidos en eventos de cada equipo (A y B)
Separa los eventos en lineups

psl_equipo_A = 0

Para cada Lineup de un Equipo en un Partido

    Encontrar para cada jugador (pi) del lineup:
        - Tiempo jugado (Propio del Lineup en Minutos) t(pi, pi)
        - Cantidad de Gains (Veces que recupero la posesion)
        - Cantidad de Loss (Veces que perdio la posesion)
        - Cantidad de Tiros

        Por cada compañero en el lineup (pj):

            - Tiempo compartido (Propio del Lineup en Minutos) t(pi, pj)
            - Cantidad de Pases de pi a pj

        - Calcular transition rates
            - r(G, pi) = Gains / t(pi, pi)
            - r(pi, L) = Loss / t(pi, pi)
            - r(pi, S) = Tiros / t(pi, pi)
            - r(pi, pj) = Pases(pi, pj) / t(pi, pi) \forall pj \in Lineup
        
        - Calcular el coeficiente ω (Formula (17) página 12)

        - Con el coeficiente ω, actualizamos r(pi,pj) y r(pi,S)
        -rω(pi,pj)=ω r(pi,pj), rω(pi,S)=ω r(pi,S)

        - Calcular las qs
                                           rω(U,V)
        qω(U, V) =  -------------------------------------------------------------           (8)
                     rω(U,G) + rω(U,S) + rω(U,L) + sum_{i=1}^{11}{rω(U,p_i)} + 1


    Armar la matriz de adyacencia de pases Q

             | 0 qω(G,   p_1) qω(G,   p_2) ... qω(G,   p_11)      0           0      |
             | 0 qω(p_1, p_1) qω(p_1, p_2) ... qω(p_1, p_11) qω(p_1, S) qω(p_1, L) |
    Qω(A) = |               ⋮                                                          |    (9)
             | 0 qω(p_11,p_1) qω(p_11,p_2) ... qω(p_11,p_11) qω(p_11,S) qω(p_11,L) |
             | 0        0            0       ...       0             1           0      |
             | 0        0            0       ...       0             0           1      |

    A partir de Qω(A) calcular el PSL del lineup

    psl_estimator(Q: np.array) -> float

    PSL_lineup = psl_estimator(Q_g(A))

    Incrementar psl_equipo_A += PSL_lineup * tiempo del lineup

    psl_equipo_A += PSL_lineup * tiempo del lineup

psl_equipo_A = psl_equipo_A / tiempo total del partido

# Para la parte predictiva

En el caso que mantengamos el r(G, pi), r(pi, S) y r(pi, L) de cada jugador (nuevo) como intrinsecos al jugador y no es afectado por el equipo:

    Para cada par de Jugadores pi y pj que jugaron juntos, guardar sus r(pi, pj)
    Estimar r^(pi, pj)

    Para el encontrar el mejor jugador para tu equipo
    Agarramos otros jugadores que no esten en tu equipo
    Calculamos el PSL del equipo normal y el PSL del equipo con el jugador nuevo (reemplazando a uno del equipo)
    Rankeamos los jugadores por el aporte de PSL (Ver Tabla 3 - Página 24)

Si no:
    Previo a calcular el PSL del equipo
    Estimar r(G, pi), r(pi, S) y r(pi, L) para el nuevo jugador pi teniendo en cuenta el equipo.

# Funciones a implementar

```python

Evento = pd.Series
Partido = pd.DataFrame
Temporada = pd.DataFrame # Eventos de todos los partidos de una temporada
EventosLineup = pd.DataFrame # Eventos de un lineup en un partido
Jugador = str
TransitionMatrix = np.array # Con forma (14, 14), la suma de cada fila es 1

def leer_excel(path: str) -> Temporada:
    pass

# Bloque Spliting

def separar_partidos(df: Temporada) -> List[Partido]:
    pass

def separar_partido_en_equipo(df: Partido) -> Tuple[Partido, Partido]:
    pass

def separar_partido_del_equipo_en_lineups(df: Partido) -> List[EventosLineup]:
    pass



def get_jugadores(lineup: EventosLineup) -> List[Jugador]:
    pass

def get_passes(lineup: EventosLineup, jugador: Jugador) -> int:
    pass

def get_gains(lineup: EventosLineup, jugador: Jugador) -> int:
    pass

def get_loss(lineup: EventosLineup, jugador: Jugador) -> int:
    pass

def get_shots(lineup: EventosLineup, jugador: Jugador) -> int:
    pass

def get_time_played(lineup: EventosLineup, jugador: Jugador) -> int:
    # Propio del lineup
    pass

def get_shared_time(lineup: EventosLineup, jugador: Jugador, jugador2: Jugador) -> int:
    # Propio del lineup
    pass

def get_lineup_duration(lineup: EventosLineup) -> int:
    pass

def get_ratio_gains(lineup: EventosLineup, jugador: Jugador) -> float:
    pass

def get_ratio_loss(lineup: EventosLineup, jugador: Jugador) -> float:
    pass

def get_ratio_shots(lineup: EventosLineup, jugador: Jugador) -> float:
    pass

def get_ratio_passes(lineup: EventosLineup, jugador: Jugador, jugador2: Jugador) -> float:
    pass

def build_R(lineup: EventosLineup) -> TransitionMatrix:
    pass

def build_Q(R: TransitionMatrix) -> TransitionMatrix:
    # Normaliza R
    pass

def psl_estimator(Q: np.array) -> float:
    pass

def psl_equipo_partido(equipo, partido) -> float:
    # sum psl_lineup * tiempo_lineup / tiempo_total_partido
    pass
```
