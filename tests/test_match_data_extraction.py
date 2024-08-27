# pylint: disable=W0613
# pylint: disable=C0103
# pylint: disable=C0301

""" This module contains tests for the match data extraction module. """

import unittest

import numpy as np
from src.event_processing import (
    leer_excel,
    separar_partido_del_equipo_en_lineups,
    separar_partido_en_equipo_pov,
    separar_partidos,
    # separar_partidos,
    # separar_partido_en_equipo_pov,
)

from src.match_data_extraction import get_jugadores, get_lineup_duration


class TestEventProcessing(unittest.TestCase):
    """Tests for the event processing module"""

    def setUp(self):
        self.temporada = leer_excel("./SampleData/epl.xlsx")
        self.partidos = separar_partidos(self.temporada)
        self.jugadores = get_jugadores(self.temporada)

    def test_get_jugadores_type(self):
        """ Type of jugadores is List of integers """
        jugadores = self.jugadores
        self.assertIsInstance(jugadores, list)
        self.assertTrue(all(isinstance(jugador, int) for jugador in jugadores))

    def test_no_duplicates_jugadores(self):
        """ Check that there are no duplicate players in the data """
        jugadores = self.jugadores
        self.assertEqual(len(jugadores), len(set(jugadores)))

    def test_no_jugadores_not_in_data(self):
        """ Assert all players are ids present int the data """
        jugadores = self.jugadores
        self.assertTrue(all(jugador in self.temporada["player_id"].unique() for jugador in jugadores))

    def test_jugadores_in_lineup(self):
        """ Assert all players are integers """
        jugadores = self.jugadores
        self.assertTrue(all(isinstance(jugador, int) for jugador in jugadores))

    def test_lineup_length(self):
        """ Check that the length of a lineup is less than or equal to 11 """
        sample_partido = np.random.choice(self.partidos)
        sample_equipo = np.random.choice(separar_partido_en_equipo_pov(sample_partido))
        sample_lineup = np.random.choice(separar_partido_del_equipo_en_lineups(sample_equipo))
        jugadores = get_jugadores(sample_lineup)
        # Assert len <= 11
        self.assertTrue(len(jugadores) <= 11)

    def test_lineup_duration(self):
        """ Check that the duration of a lineup is not negative"""

        for partido in self.partidos:
            for equipo in separar_partido_en_equipo_pov(partido):
                for lineup in separar_partido_del_equipo_en_lineups(equipo):
                    self.assertTrue(get_lineup_duration(lineup) >= 0)

    def test_lineup_duration_less_than_match_duration(self):
        """ Check that the duration of a lineup is not greater than the duration of the match """

        for partido in self.partidos:
            for equipo in separar_partido_en_equipo_pov(partido):
                for lineup in separar_partido_del_equipo_en_lineups(equipo):
                    self.assertTrue(get_lineup_duration(lineup) <= get_lineup_duration(equipo))

    def test_player_time_played_less_than_lineup_duration(self):
        """ Check that the time played for a player is not greater than the duration of the lineup in which he played """

        for partido in self.partidos:
            for equipo in separar_partido_en_equipo_pov(partido):
                for lineup in separar_partido_del_equipo_en_lineups(equipo):
                    jugadores = get_jugadores(lineup)
                    for jugador in jugadores:
                        self.assertTrue(get_lineup_duration(lineup) >= get_lineup_duration(lineup[lineup["player_id"] == jugador]))

    def test_player_time_played(self):
        """ Get the time played by a player in a lineup """

        # Check that the time played by a player is not negative
        for partido in self.partidos:
            for equipo in separar_partido_en_equipo_pov(partido):
                for lineup in separar_partido_del_equipo_en_lineups(equipo):
                    jugadores = get_jugadores(lineup)
                    for jugador in jugadores:
                        self.assertTrue(get_lineup_duration(lineup[lineup["player_id"] == jugador]) >= 0)

    def test_player_10_players_not_in_lineup(self):
        """ Get the time played by a player not in a lineup exactly 10 times expect 0 """

        # Check that for a player not in a lineup, the time played is 0, 10 times
        for _ in range(10):

            sample_partido = np.random.choice(self.partidos)
            sample_equipo = np.random.choice(separar_partido_en_equipo_pov(sample_partido))
            sample_lineup = np.random.choice(separar_partido_del_equipo_en_lineups(sample_equipo))
            jugadores = get_jugadores(sample_lineup)

            # Sample player not in the lineup
            other_player = self.partidos[~self.partidos["player_id"].isin(jugadores)]["player_id"].sample().iloc[0]

            self.assertEqual(get_lineup_duration(sample_lineup[sample_lineup["player_id"] == other_player]), 0)
