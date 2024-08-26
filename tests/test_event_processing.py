""" This module contains tests for the event processing module. """

import unittest

import pandas as pd
from src.event_processing import (
    leer_excel,
    separar_partidos,
    separar_partido_en_equipo_pov,
    separar_partido_del_equipo_en_lineups,
)


class TestEventProcessing(unittest.TestCase):
    """Tests for the event processing module"""

    def setUp(self):
        self.temporada = leer_excel("./SampleData/epl.xlsx")

    def test_type_temporada(self):
        """ Test the type of the Temporada object """
        self.assertIsInstance(self.temporada, pd.DataFrame)

    def test_leer_excel(self):
        """ Leer un archivo Excel con eventos de partidos de futbol """
        self.assertEqual(self.temporada.shape, (648883, 22))

    def test_separar_partidos(self):
        """ Separar los eventos de una temporada en partidos individuales """
        partidos = separar_partidos(self.temporada)
        self.partidos = partidos

        # Test Type partidos is List of DataFrames
        self.assertIsInstance(partidos, list)
        self.assertTrue(all(isinstance(partido, pd.DataFrame) for partido in partidos))

        self.assertEqual(len(partidos), 380)

        for partido in partidos:
            # Cada partido tiene un solo id
            self.assertTrue(partido["match_id"].nunique() == 1)

            # Cada partido tiene exactamente 2 equipos
            self.assertTrue(partido["team_id"].nunique() == 2)

        # La suma de los eventos de todos los partidos
        # es igual a la cantidad de eventos en la temporada
        self.assertEqual(
            sum([partido.shape[0] for partido in partidos]), self.temporada.shape[0]
        )

    def test_separar_partido_en_equipo_pov(self):
        """ Separar los eventos de un partido en dos DataFrames, uno por equipo """
        partidos = separar_partidos(self.temporada)
        partido = partidos[0]
        equipos = separar_partido_en_equipo_pov(partido)

        # Test Type equipos is Tuple of DataFrames
        self.assertIsInstance(equipos, tuple)
        self.assertTrue(all(isinstance(equipo, pd.DataFrame) for equipo in equipos))

        equipo1, equipo2 = equipos
        # Assert que los equipos sean df
        self.assertIsInstance(equipo1, pd.DataFrame)
        self.assertIsInstance(equipo2, pd.DataFrame)

        self.assertEqual(equipo1.shape[0] + equipo2.shape[0], partido.shape[0])

        # Cada equipo tiene un solo id
        self.assertEqual(equipo1["team_id"].nunique(), 1)
        self.assertEqual(equipo2["team_id"].nunique(), 1)

        # Los ids de los equipos son diferentes
        self.assertNotEqual(
            equipo1["team_id"].unique()[0], equipo2["team_id"].unique()[0]
        )

    def test_separar_1_partido_del_equipo_en_lineups(self):
        """ Separar los eventos de un equipo en lineups """
        partidos = self.partidos
        partido = partidos[0]
        equipos = separar_partido_en_equipo_pov(partido)
        equipo1, _ = equipos

        lineups = separar_partido_del_equipo_en_lineups(equipo1)

        # Test Type lineups is List of DataFrames
        self.assertIsInstance(lineups, list)
        self.assertTrue(all(isinstance(lineup, pd.DataFrame) for lineup in lineups))

        # La cantidad de jugadores en cada lineup es <= 11
        for lineup in lineups:
            self.assertTrue(lineup["player_id"].nunique() <= 11)         


if __name__ == "__main__":
    unittest.main()
