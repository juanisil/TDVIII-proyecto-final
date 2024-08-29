# pylint: disable=W0613
# pylint: disable=C0103
# pylint: disable=C0301

""" Módulo para armar Dataset """

from typing import Dict
import numpy as np
import pandas as pd
# import os
# import pandas as pd

from src.event_processing import leer_excel, separar_partido_del_equipo_en_lineups, separar_partido_en_equipo_pov, separar_partidos
from src.match_data_extraction import get_jugadores
from src.utils_CMTC import get_ratio_passes


def make_features(player) -> np.array:
    """ Retorna un vector de features para un jugador

    Args:
        player
    """

    return np.array([])


class Dataset:
    """
        Clase para armar un Dataset de pases entre jugadores de futbol
    """

    def __init__(self, path):
        self.path = path
        self.raw_data = leer_excel(path)

        self.features = []
        self.target = []

        for partido in separar_partidos(self.raw_data):
            for equipo in separar_partido_en_equipo_pov(partido):
                for lineup in separar_partido_del_equipo_en_lineups(equipo):
                    players = get_jugadores(lineup)

                    for player_1 in players:
                        for player_2 in players:
                            if player_1 != player_2:
                                self.features.append(
                                    [
                                        player_1
                                        ** make_features(player_1),
                                        player_2
                                        ** make_features(player_2),
                                    ]
                                )
                                self.target.append(
                                    [
                                        get_ratio_passes(lineup, player_1, player_2),
                                    ]
                                )

        self.columns = ["player_1", "player_2"]
        self.features = np.array(self.features)
        self.target = np.array(self.target)

    def to_DataFrame(self) -> pd.DataFrame:
        """ Retorna un DataFrame con los datos del dataset """
        return pd.DataFrame(
            np.concatenate([self.features, self.target], axis=1), columns=self.columns + ["target"]
        )

    def get_data(self) -> Dict[str, np.array]:
        """ Get the features and target arrays

        Returns:
            dict: Dictionary with the features and target arrays
        """
        return {
            "features": self.features,
            "target": self.target
        }

    def __len__(self) -> int:
        """ Get the length of the dataset """
        return len(self.features)

    def __getitem__(self, idx):
        """ Get an item from the dataset """
        return self.features[idx], self.target[idx]

    def __iter__(self):
        """ Iterate over the dataset """

        # Zip the features and target arrays together
        data = zip(self.features, self.target)
        for x, y in data:
            yield x, y
