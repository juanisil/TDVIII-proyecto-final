.. Cómo encontrar el mejor jugador para tu Equipo de Fútbol documentation master file, created by
   sphinx-quickstart on Wed Nov 27 14:42:49 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Cómo encontrar el mejor jugador para tu Equipo de Fútbol documentation
======================================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: src
   :members:

.. autofunction:: src.event_processing.leer_excel
.. autofunction:: src.event_processing.separar_partidos
.. autofunction:: src.event_processing.separar_partido_en_equipo_pov
.. autofunction:: src.event_processing.separar_partido_del_equipo_en_lineups

.. autofunction:: src.excel_xml_util.change_in_zip

.. autoclass:: src.futbol_types.Evento
.. autoclass:: src.futbol_types.Partido
.. autoclass:: src.futbol_types.Temporada
.. autoclass:: src.futbol_types.EventosLineup
.. autoclass:: src.futbol_types.Jugador
.. autoclass:: src.futbol_types.TransitionMatrix

.. autofunction:: src.match_data_extraction.get_jugadores
.. autofunction:: src.match_data_extraction.get_passes
.. autofunction:: src.match_data_extraction.get_gains
.. autofunction:: src.match_data_extraction.get_losses
.. autofunction:: src.match_data_extraction.get_shots
.. autofunction:: src.match_data_extraction.get_time_played
.. autofunction:: src.match_data_extraction.get_shared_time
.. autofunction:: src.match_data_extraction.get_lineup_duration

.. autofunction:: src.utils_CTMC.get_ratio_gains
.. autofunction:: src.utils_CTMC.get_ratio_loss
.. autofunction:: src.utils_CTMC.get_ratio_shots
.. autofunction:: src.utils_CTMC.get_ratio_passes
.. autofunction:: src.utils_CTMC.build_R
.. autofunction:: src.utils_CTMC.build_Q
.. autofunction:: src.utils_CTMC.psl_estimator

.. autoclass:: src.Player2Vec.Player2Vec
.. autoclass:: src.Player2Vec.EPL_Graph

.. autofunction:: src.p2v_dist.p2v_dist_model
.. autofunction:: src.p2v_dist.custom_loss
.. autofunction:: src.p2v_dist.JSD
