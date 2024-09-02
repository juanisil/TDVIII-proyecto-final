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

from src.match_data_extraction import get_jugadores, get_shared_time, get_gains, get_losses, get_passes, get_shots, get_goals


class TestEventProcessing(unittest.TestCase):
    """Tests for the event processing module"""

    def setUp(self):
        self.temporada = leer_excel("./SampleData/epl_test.xlsx")
        self.partido = separar_partidos(self.temporada)[0]
        self.eventos_arsenal, self.eventos_sunder = separar_partido_en_equipo_pov(self.partido)
        self.lineup_arsenal = separar_partido_del_equipo_en_lineups(self.eventos_arsenal)[0]
        self.lineup_sunder = separar_partido_del_equipo_en_lineups(self.eventos_sunder)[0]



    def test_get_losses(self):
         #   	Losses	
        # team		id    Losses
        #Arsenal	20467	2
        #Sunder	    15073	2
        #Arsenal	80254	1
        #Sunder	    63370	1


        self.assertEqual(get_losses(self.lineup_arsenal, 20467), 2)
        self.assertEqual(get_losses(self.lineup_sunder, 15073), 2)
        self.assertEqual(get_losses(self.lineup_arsenal, 80254), 1)
        self.assertEqual(get_losses(self.lineup_sunder, 63370), 1)
       

    def test_get_gains(self):
            #Gains
        # team		id    Gains		
        #34392	1	Sunder
        #80254	2	Arsenal
        #63370	1	Sunder
        #15073	1	Sunder
        #8758	1	Arsenal

        self.assertEqual(get_gains(self.lineup_sunder, 34392), 1)
        self.assertEqual(get_gains(self.lineup_arsenal, 80254), 2)
        self.assertEqual(get_gains(self.lineup_sunder, 63370), 1)
        self.assertEqual(get_gains(self.lineup_sunder, 15073), 1)
        self.assertEqual(get_gains(self.lineup_arsenal, 8758), 1)

    def test_get_passes(self):
        	#Pases	
        #ID1    ID2	Pases Team 	
        #17733	19524	1	Arsenal
        #19524	15943	1	Arsenal
        #15943	20467	1	Arsenal
        #34392	15073	1	Sunder
        #80254	8758	1	Arsenal
        #8758	20467	1	Arsenal
        #8758	80254	1	Arsenal
        #80254	15943	1	Arsenal
        #15943	42427	1	Arsenal
    
        self.assertEqual(get_passes(self.lineup_arsenal, 17733, 19524), 1)
        self.assertEqual(get_passes(self.lineup_arsenal, 19524, 15943), 1)
        self.assertEqual(get_passes(self.lineup_arsenal, 15943, 20467), 1)
        self.assertEqual(get_passes(self.lineup_sunder, 34392, 15073), 1)
        self.assertEqual(get_passes(self.lineup_arsenal, 80254, 8758), 1)
        self.assertEqual(get_passes(self.lineup_arsenal, 8758, 20467), 1)
        self.assertEqual(get_passes(self.lineup_arsenal, 8758, 80254), 1)
        self.assertEqual(get_passes(self.lineup_arsenal, 80254, 15943), 1)
        self.assertEqual(get_passes(self.lineup_arsenal, 15943, 42427), 1)

    def test_get_shots(self):
        #Tiros
        #ID	Tiros
        #42427	1   Arsenal

        self.assertEqual(get_shots(self.lineup_arsenal, 42427), 1)


    def test_get_goals(self):
        #Tiros
        #ID	Tiros
        #42427	1   Arsenal

        self.assertEqual(get_goals(self.lineup_arsenal, 42427), 1)

    
    def test_get_jugadores(self):
        #Arsenal
        #ID
        #8758
        #19524
        #42427
        #80254
        #17733
        #15943
        #20467

        self.assertEqual(set(get_jugadores(self.lineup_arsenal)), set([8758, 19524, 42427, 80254, 17733, 15943, 20467]))


    def test_get_shared_time(self):
        #ID1    ID2	Tiempo compartido
        #17733	19524	33 segs

        self.assertEqual(get_shared_time(self.lineup_arsenal, 17733, 19524), 33/60)

