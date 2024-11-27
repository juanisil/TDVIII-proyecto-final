<!-- Cómo encontrar el mejor jugador para tu Equipo de Fútbol documentation master file, created by
sphinx-quickstart on Wed Nov 27 14:42:49 2024.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive. -->

# Cómo encontrar el mejor jugador para tu Equipo de Fútbol documentation

<a id="module-src"></a>

### src.event_processing.leer_excel(path: str) → [Temporada](#src.futbol_types.Temporada)

Lee un archivo Excel con eventos de partidos de futbol.

Args:
: path (str): Ruta al archivo Excel.

Returns:
: Temporada: Eventos de todos los partidos de una temporada.

Columns: [“season_id”, “match_id”, “home_team_id”, “home_team_name”,
: “away_team_id”, “away_team_name”, “id”, “event_id”, “date”, “time”,
  “period_id”, “min”, “sec”, “team_id”, “player_id”, “playerName”,
  “playerPosition”, “x”, “y”, “type”, “description”, “outcome”].

### src.event_processing.separar_partidos(df: [Temporada](#src.futbol_types.Temporada)) → List[[Partido](#src.futbol_types.Partido)]

Separa los eventos de una temporada en partidos individuales

Args:
: df (Temporada): Eventos de todos los partidos de una temporada

Returns:
: List[Partido]: Eventos de cada partido

### src.event_processing.separar_partido_en_equipo_pov(df: [Partido](#src.futbol_types.Partido)) → Tuple[[Partido](#src.futbol_types.Partido), [Partido](#src.futbol_types.Partido)]

Separa los eventos de un partido en dos DataFrames, uno por equipo

Args:
: df (Partido): Eventos de un partido

Returns:
: Tuple[Partido, Partido]: Eventos de cada equipo

### src.event_processing.separar_partido_del_equipo_en_lineups(df: [Partido](#src.futbol_types.Partido)) → List[[EventosLineup](#src.futbol_types.EventosLineup)]

Separa los eventos de un equipo en dos DataFrames, uno por lineup

Args:
: df (Partido): Eventos de un equipo en un partido

Returns:
: List[EventosLineup]: Eventos de cada lineup

### src.excel_xml_util.change_in_zip(file_name: str, name_filter: str, change: callable)

Fixer for the «synchVertical» property in the Excel files

Args:
: file_name (str): Path to the Excel file
  name_filter (str): Filter for the files to change
  change (callable): Function to change the data

### *class* src.futbol_types.Evento

NewType creates simple unique types with almost zero
runtime overhead. NewType(name, tp) is considered a subtype of tp
by static type checkers. At runtime, NewType(name, tp) returns
a dummy callable that simply returns its argument. Usage:

```default
UserId = NewType('UserId', int)

def name_by_id(user_id: UserId) -> str:
    ...

UserId('user')          # Fails type check

name_by_id(42)          # Fails type check
name_by_id(UserId(42))  # OK

num = UserId(5) + 1     # type: int
```

alias de `Series`

### *class* src.futbol_types.Partido

NewType creates simple unique types with almost zero
runtime overhead. NewType(name, tp) is considered a subtype of tp
by static type checkers. At runtime, NewType(name, tp) returns
a dummy callable that simply returns its argument. Usage:

```default
UserId = NewType('UserId', int)

def name_by_id(user_id: UserId) -> str:
    ...

UserId('user')          # Fails type check

name_by_id(42)          # Fails type check
name_by_id(UserId(42))  # OK

num = UserId(5) + 1     # type: int
```

alias de `DataFrame`

### *class* src.futbol_types.Temporada

NewType creates simple unique types with almost zero
runtime overhead. NewType(name, tp) is considered a subtype of tp
by static type checkers. At runtime, NewType(name, tp) returns
a dummy callable that simply returns its argument. Usage:

```default
UserId = NewType('UserId', int)

def name_by_id(user_id: UserId) -> str:
    ...

UserId('user')          # Fails type check

name_by_id(42)          # Fails type check
name_by_id(UserId(42))  # OK

num = UserId(5) + 1     # type: int
```

alias de `DataFrame`

### *class* src.futbol_types.EventosLineup

NewType creates simple unique types with almost zero
runtime overhead. NewType(name, tp) is considered a subtype of tp
by static type checkers. At runtime, NewType(name, tp) returns
a dummy callable that simply returns its argument. Usage:

```default
UserId = NewType('UserId', int)

def name_by_id(user_id: UserId) -> str:
    ...

UserId('user')          # Fails type check

name_by_id(42)          # Fails type check
name_by_id(UserId(42))  # OK

num = UserId(5) + 1     # type: int
```

alias de `DataFrame`

### *class* src.futbol_types.Jugador

NewType creates simple unique types with almost zero
runtime overhead. NewType(name, tp) is considered a subtype of tp
by static type checkers. At runtime, NewType(name, tp) returns
a dummy callable that simply returns its argument. Usage:

```default
UserId = NewType('UserId', int)

def name_by_id(user_id: UserId) -> str:
    ...

UserId('user')          # Fails type check

name_by_id(42)          # Fails type check
name_by_id(UserId(42))  # OK

num = UserId(5) + 1     # type: int
```

alias de `str`

### *class* src.futbol_types.TransitionMatrix

NewType creates simple unique types with almost zero
runtime overhead. NewType(name, tp) is considered a subtype of tp
by static type checkers. At runtime, NewType(name, tp) returns
a dummy callable that simply returns its argument. Usage:

```default
UserId = NewType('UserId', int)

def name_by_id(user_id: UserId) -> str:
    ...

UserId('user')          # Fails type check

name_by_id(42)          # Fails type check
name_by_id(UserId(42))  # OK

num = UserId(5) + 1     # type: int
```

alias de `ndarray`

### src.match_data_extraction.get_jugadores(lineup: [EventosLineup](#src.futbol_types.EventosLineup)) → List[[Jugador](#src.futbol_types.Jugador)]

Obtiene los jugadores de un lineup

Args:
: lineup (EventosLineup): Eventos de un lineup

Returns:
: List[Jugador]: player_id de los jugadores del lineup

### src.match_data_extraction.get_passes(lineup: [EventosLineup](#src.futbol_types.EventosLineup), jugador: [Jugador](#src.futbol_types.Jugador), jugador2: [Jugador](#src.futbol_types.Jugador)) → int

Obtiene la cantidad de pases de un jugador a otro en un lineup

Args:
: lineup (EventosLineup): Eventos de un lineup
  jugador (Jugador): Jugador que realiza el pase
  jugador2 (Jugador): Jugador que recibe el pase

Returns:
: int: Cantidad de pases

### src.match_data_extraction.get_gains(lineup: [EventosLineup](#src.futbol_types.EventosLineup), jugador: [Jugador](#src.futbol_types.Jugador)) → int

Obtiene la cantidad de posesiones ganadas por un jugador en un lineup

Args:
: lineup (EventosLineup): Eventos de un lineup
  jugador (Jugador): Jugador que gana una posesión

Returns:
: int: Cantidad de posesiones ganadas

### src.match_data_extraction.get_losses(lineup: [EventosLineup](#src.futbol_types.EventosLineup), jugador: [Jugador](#src.futbol_types.Jugador)) → int

Obtiene la cantidad de posesiones perdidas por un jugador en un lineup

Args:
: lineup (EventosLineup): Eventos de un lineup
  jugador (Jugador): Jugador que pierde una posesión

Returns:
: int: Cantidad de posesiones perdidas

### src.match_data_extraction.get_shots(lineup: [EventosLineup](#src.futbol_types.EventosLineup), jugador: [Jugador](#src.futbol_types.Jugador), OffTarget: bool = False) → int

Obtiene la cantidad de tiros de un jugador en un lineup

Args:
: lineup (EventosLineup): Eventos de un lineup
  jugador (Jugador): Jugador que realiza el tiro
  OffTarget (bool): Si se incluyen tiros fuera de puerta

Returns:
: int: Cantidad de tiros

### src.match_data_extraction.get_time_played(lineup: [EventosLineup](#src.futbol_types.EventosLineup), jugador: [Jugador](#src.futbol_types.Jugador)) → int

Obtiene el tiempo jugado por un jugador en un lineup

Args:
: lineup (EventosLineup): Eventos de un lineup
  jugador (Jugador): Jugador

Returns:
: int: Tiempo jugado en minutos

### src.match_data_extraction.get_shared_time(lineup: [EventosLineup](#src.futbol_types.EventosLineup), jugador: [Jugador](#src.futbol_types.Jugador), jugador2: [Jugador](#src.futbol_types.Jugador)) → int

Obtiene el tiempo jugado por dos jugadores en un lineup

Args:
: lineup (EventosLineup): Eventos de un lineup
  jugador (Jugador): Jugador
  jugador2 (Jugador): Jugador

Returns:
: int: Tiempo jugado en minutos

### src.match_data_extraction.get_lineup_duration(lineup: [EventosLineup](#src.futbol_types.EventosLineup)) → int

Obtiene la duración de un lineup

Args:
: lineup (EventosLineup): Eventos de un lineup

Returns:
: int: Duración en minutos

### src.utils_CTMC.get_ratio_gains(lineup: [EventosLineup](#src.futbol_types.EventosLineup), jugador: [Jugador](#src.futbol_types.Jugador)) → float

Calcula la tasa de posesiones ganadas por un jugador en un lineup

Args:
: lineup (EventosLineup): Eventos de un lineup
  jugador (Jugador): Jugador que gana posesiones

Returns:
: r G p1 (float): Tasa de posesiones ganadas

### src.utils_CTMC.get_ratio_loss(lineup: [EventosLineup](#src.futbol_types.EventosLineup), jugador: [Jugador](#src.futbol_types.Jugador)) → float

Calcula la tasa de posesiones perdidas por un jugador en un lineup

Args:
: lineup (EventosLineup): Eventos de un lineup
  jugador (Jugador): Jugador que pierde posesiones

Returns:
: r p1 L (float): Tasa de posesiones perdidas

### src.utils_CTMC.get_ratio_shots(lineup: [EventosLineup](#src.futbol_types.EventosLineup), jugador: [Jugador](#src.futbol_types.Jugador)) → float

Calcula la tasa de tiros de un jugador en un lineup

Args:
: lineup (EventosLineup): Eventos de un lineup
  jugador (Jugador): Jugador que realiza tiros

Returns:
: r pi S (float): Tasa de tiros

### src.utils_CTMC.get_ratio_passes(lineup: [EventosLineup](#src.futbol_types.EventosLineup), jugador: [Jugador](#src.futbol_types.Jugador), jugador2: [Jugador](#src.futbol_types.Jugador)) → float

Calcula la tasa de pases de un jugador a otro en un lineup

Args:
: lineup (EventosLineup): Eventos de un lineup
  jugador (Jugador): Jugador que realiza el pase
  jugador2 (Jugador): Jugador que recibe el pase

Returns:
: r pi, pj (float): Tasa de pases

### src.utils_CTMC.build_R(lineup: [EventosLineup](#src.futbol_types.EventosLineup)) → [TransitionMatrix](#src.futbol_types.TransitionMatrix)

Construye la matriz de transición de estados R (Como Q pero sin Normalización) de un lineup

Args:
: lineup (EventosLineup): Eventos de un lineup

Returns:
: R (TransitionMatrix): Matriz de transición de estados sin normalizar

### src.utils_CTMC.build_Q(R: [TransitionMatrix](#src.futbol_types.TransitionMatrix)) → [TransitionMatrix](#src.futbol_types.TransitionMatrix)

Dada una matriz de transición de estados R,
: construye la matriz de transición de estados Q normalizando R

Args:
: R (TransitionMatrix): Matriz de transición de estados sin normalizar

Returns:
: Q (TransitionMatrix): Matriz de transición de estados normalizada

### src.utils_CTMC.psl_estimator(Q: [TransitionMatrix](#src.futbol_types.TransitionMatrix)) → float

Calcula la probabilidad de perder la pelota
: antes de tirar al arco de un lineup
  a partir de la matriz de transición de estados Q

Args:
: Q (TransitionMatrix): Matriz de transición de estados normalizada

Returns:
: PSL (float): Probabilidad de perder la pelota antes de tirar al arco

### *class* src.Player2Vec.Player2Vec(model_path=None, epl_data=None)

Class to train and use a Node2Vec model over the EPL player graph

### *class* src.Player2Vec.EPL_Graph(epl_data: EPL_Data)

Class to build the EPL player graph

### src.p2v_dist.p2v_dist_model(input_size=3, output_size=10)

Player2Vec to Player Stats Distribution Model

### src.p2v_dist.custom_loss()

Custom loss function to calculate the Jensen-Shannon Divergence between the predicted and target distributions for the 5 features
Ponderation of the JSD divergence of the mean and std of the 5 features

### src.p2v_dist.JSD()

Jenson-Shannon Divergence Loss Function

[https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/13](https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/13)
