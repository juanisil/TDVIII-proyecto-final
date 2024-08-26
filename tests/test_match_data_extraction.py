""" This module contains tests for the match data extraction module. """

import unittest
from src.event_processing import (
    leer_excel,
    # separar_partidos,
    # separar_partido_en_equipo_pov,
)

from src.match_data_extraction import get_jugadores


class TestEventProcessing(unittest.TestCase):
    """Tests for the event processing module"""

    def setUp(self):
        self.temporada = leer_excel("./SampleData/epl.xlsx")

    def test_get_jugadores(self):
        """ Get the players from a lineup """
        jugadores = get_jugadores(self.temporada)
        # Assert no duplicates
        self.assertEqual(len(jugadores), len(set(jugadores)))

        # Assert all players are ids present int the data
        self.assertTrue(all(jugador in self.temporada["player_id"].unique() for jugador in jugadores))
