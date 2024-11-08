"""Bayesian PSL Player Comparison"""

# pylint: disable=E1126
# pylint: disable=E0401
# pylint: disable=C0413
# pylint: disable=C0411
# pylint: disable=C0103
# pylint: disable=C0301
# pylint: disable=W0612

from src.utils_CTMC import build_Q, psl_estimator  # noqa: E402
from src.match_data_extraction import (  # noqa: E402
    get_jugadores,
    get_lineup_duration,
)

from src.epl_player_data_utils import EPLPlayerData  # noqa: E402
from src.futbol_types import Partido  # noqa: E402

from src.event_processing import (  # noqa: E402
    leer_excel,
    separar_partido_del_equipo_en_lineups,
    separar_partido_en_equipo_pov,
    separar_partidos,
)

from tqdm import tqdm  # noqa: E402

import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from typing import List


class EPL_Data:
    """Class to Encapsulate the different sources of EPL data"""

    def __init__(self, epl_path, players_path, r_storage_path):
        self.epl_path = epl_path
        self.players_path = players_path
        self.r_storage_path = r_storage_path

        self.epl = leer_excel(self.epl_path)
        self.epl_player_data = EPLPlayerData(self.players_path)
        self.R_storage = np.load(self.r_storage_path)

        self.partidos = separar_partidos(self.epl)

        self.Q_storage = self.Q_storage_from_R_storage()

        self.tp_ds = None
        self.build_transition_prob_dataset()

        self.player_ids = self.get_transition_prob_dataset()["player_id"].unique()

        self.player_kdes = RandomVariablePSL.calculate_player_kdes(
            self.tp_ds, self.player_ids
        )

        self.size = 0

        self.calculate_size()

    def get_epl(self):
        """Getter for the EPL DataFrame

        Returns:
            pd.DataFrame: EPL DataFrame
        """

        return self.epl

    def get_epl_player_data(self):
        """Getter for the EPL Player Data

        Returns:
            EPLPlayerData: EPL Player Data
        """

        return self.epl_player_data

    def get_R_storage(self) -> np.ndarray:
        """Getter for the R Storage

        Returns:
            np.ndarray: R Storage
        """

        return self.R_storage

    def Q_storage_from_R_storage(self, R_storage=None):
        """Build Q Storage from R Storage

        Args:
            R_storage (np.ndarray, optional): R Storage. Defaults to None.

        Returns:
            np.ndarray: Q Storage
        """

        if R_storage is None:
            R_storage = self.R_storage
        Q_storage = np.zeros(R_storage.shape)
        for pi in range(R_storage.shape[0]):
            for ti in range(R_storage.shape[1]):
                for li in range(R_storage.shape[2]):
                    Q_storage[pi, ti, li, 1:, 1:] = build_Q(
                        R_storage[pi, ti, li, 1:, 1:]
                    )
                    Q_storage[pi, ti, li, 0, :] = R_storage[pi, ti, li, 0, :]
                    Q_storage[pi, ti, li, :, 0] = R_storage[pi, ti, li, :, 0]
        return Q_storage

    def get_Q_storage(self) -> np.ndarray:
        """Getter for the Q Storage

        Returns:
            np.ndarray: Q Storage
        """

        return self.Q_storage

    def get_partidos(self) -> List[Partido]:
        """Getter for the Partidos

        Returns:
            List[Partido]: List of Partidos
        """

        return self.partidos

    def get_transition_prob_dataset(self) -> pd.DataFrame:
        """Getter for the Transition Probability Dataset

        Returns:
            pd.DataFrame: Transition Probability Dataset
        """

        return self.tp_ds

    def get_player_total_duration(self, player_id):
        """Getter for the Player Total Duration

        Args:
            player_id (int): Player ID

        Returns:
            float: Player Total Duration
        """

        return self.tp_ds[self.tp_ds["player_id"] == player_id]["duration"].sum()

    def get_player_ids(self):
        """Getter for the Player IDs

        Returns:
            List[int]: List of Player IDs
        """

        return self.player_ids

    def get_all_player_names(self):
        """Getter for all the player names in the EPL

        Returns:
            List[str]: List of Player Names
        """

        return filter(
            lambda x: x is not None,
            [
                self.epl_player_data.get_player_name(int(player_id))
                for player_id in self.player_ids
            ]
        )

    def get_player_names_for_team(self, team_name):
        """Getter for all the player names in the EPL

        Returns:
            List[str]: List of Player Names
        """
        team_id = self.epl[self.epl["home_team_name"] == team_name][
            "home_team_id"
        ].values[0]
        team_players_ids = (
            self.epl[self.epl["team_id"] == team_id]["player_id"].dropna().unique()
        )

        return filter(
            lambda x: x is not None,
            [
                self.epl_player_data.get_player_name(int(player_id))
                for player_id in team_players_ids
            ]
        )

    def get_all_teams(self):
        """Getter for all the teams in the EPL

        Returns:
            List[str]: List of Team Names
        """

        return self.epl["home_team_name"].unique()

    def get_player_team(self, player_id):
        """Get the team of a player

        Args:
            player_id (int): Player ID

        Returns:
            str: Team Name
        """

        for pi, partido in enumerate(self.partidos):
            for ti, equipo in enumerate(separar_partido_en_equipo_pov(partido)):
                if player_id in equipo["player_id"].unique():
                    team_id = equipo["team_id"].values[0]
                    return self.epl[self.epl["home_team_id"] == team_id][
                        "home_team_name"
                    ].values[0]
        return None

    def get_player_kdes(self):
        """Getter for the Player KDEs

        Returns:
            Dict[int, Dict[str, gaussian_kde]]: Dictionary with player IDs as keys and dictionaries of KDEs as values
        """

        return self.player_kdes

    def build_transition_prob_dataset(self):
        """Build the Transition Probability Dataset"""

        trans_prob_dataset = []

        for pi, partido in enumerate(self.partidos):
            match_id = partido["match_id"].values[0]
            for ti, equipo in enumerate(separar_partido_en_equipo_pov(partido)):
                for li, lineup in enumerate(
                    separar_partido_del_equipo_en_lineups(equipo)
                ):
                    # R = R_storage[pi, ti, li, 1:, 1:]
                    Q = self.Q_storage[pi, ti, li, 1:, 1:]
                    players = self.R_storage[pi, ti, li, 0, :]
                    for player_id in players:
                        player_pos = np.where(players == player_id)[0][0]

                        # Value in R[player_pos, 13] is the ratio of shots to time played for player_id
                        p_data = {
                            "player_id": player_id,
                            "team_id": equipo["team_id"].values[0],
                            "team_index": ti,
                            "match_id": match_id,
                            "match_num": pi,
                            "lineup_index": li,
                            "duration": get_lineup_duration(lineup),
                            "gains_prob": Q[player_pos, 1],
                            "losses_prob": Q[player_pos, 12],
                            "shots_prob": Q[player_pos, 13],
                            "avg_pass_to_prob": Q[player_pos, 1:12].mean(),
                            "avg_pass_from_prob": Q[1:12, player_pos].mean(),
                            **{
                                f"pass_to_{i}": Q[1 + player_pos, i + 1]
                                for i in range(11)
                            },
                            **{
                                f"pass_from_{i}": Q[1 + i, player_pos + 1]
                                for i in range(11)
                            },
                        }
                        trans_prob_dataset.append(p_data)

        self.tp_ds = pd.DataFrame(trans_prob_dataset)

    def __iter__(self):
        for pi, partido in enumerate(self.partidos):
            match_id = partido["match_id"].values[0]
            for ti, equipo in enumerate(separar_partido_en_equipo_pov(partido)):
                team_id = equipo["team_id"].values[0]
                for li, lineup in enumerate(
                    separar_partido_del_equipo_en_lineups(equipo)
                ):
                    yield pi, ti, li, lineup, match_id, team_id

    def calculate_size(self):
        """Calculate the Size of the Dataset"""

        self.size = 0
        for _ in self:
            self.size += 1

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"{self.__class__.__name__}({self.epl_path}, {self.players_path}, {self.r_storage_path}), {len(self.partidos)} Partidos, {len(self.player_ids)} Jugadores"


class RandomVariablePSL:
    """Class for Random Variable PSL"""

    @staticmethod
    def calculate_player_kdes(tp_ds, player_ids):
        """Calculate KDEs for each player in the dataset

        Args:
            tp_ds (pd.DataFrame): Transition Probability Dataset
            player_ids (List[int]): List of player IDs

        Returns:
            Dict[int, Dict[str, gaussian_kde]]: Dictionary with player IDs as keys and dictionaries of KDEs as values
        """

        player_kdes = {}
        for player_id in player_ids:
            player_kdes[player_id] = {
                prob: (
                    gaussian_kde(probs[probs > 0])
                    if (probs := tp_ds[tp_ds["player_id"] == player_id][prob]).shape[0]
                    > 3
                    and probs[probs > 0].shape[0] > 3
                    else 0
                )
                for prob in [
                    "losses_prob",
                    "gains_prob",
                    "shots_prob",
                    "avg_pass_to_prob",
                    "avg_pass_from_prob",
                ]
            }
        return player_kdes

    @staticmethod
    def probability_psl_greater_than(psls, psls_2, n_kde_samples=10000):
        """Calculate the probability that the PSL of the first dataset is greater than the PSL of the second dataset

        Args:
            psls (List[float]): List of PSL values
            psls_2 (List[float]): List of PSL values for the second dataset
            n_kde_samples (int, optional): Number of samples to use for the KDE. Defaults to 10000.

        Returns:
            float: Probability that the PSL of the first dataset is greater than the PSL of the second
        """

        psl_kde_1 = gaussian_kde(psls)
        psl_kde_2 = gaussian_kde(psls_2)

        psl_1_samples = psl_kde_1.resample(n_kde_samples)[0]
        psl_2_samples = psl_kde_2.resample(n_kde_samples)[0]

        p_greater = np.mean(psl_1_samples > psl_2_samples)

        return p_greater

    @staticmethod
    def plot_psl_distributions(psls, psls_2, ax=None, names=None, colors=None):
        """Plot the PSL distributions of two datasets

        Args:
            psls (List[float]): List of PSL values
            psls_2 (List[float]): List of PSL values for the second dataset
        """

        if names is None:
            names = ["P1", "P2"]

        if colors is None:
            colors = ["b", "r"]

        psl_kde = gaussian_kde(psls)
        psl_kde_2 = gaussian_kde(psls_2)

        x = np.linspace(0, 1, 1000)
        psl_kde_values = psl_kde.evaluate(x)
        psl_kde_2_values = psl_kde_2.evaluate(x)

        p_greater = RandomVariablePSL.probability_psl_greater_than(psls_2, psls)

        if ax is None:
            _, ax = plt.subplots(figsize=(20, 8))

        ax.fill_between(
            x,
            psl_kde_values,
            psl_kde_2_values,
            where=(psl_kde_values > psl_kde_2_values),
            color=colors[0],
            alpha=0.5,
            label=f"psl_kde_{names[0]} > psl_kde_{names[1]}",
        )

        ax.fill_between(
            x,
            psl_kde_values,
            psl_kde_2_values,
            where=(psl_kde_values <= psl_kde_2_values),
            color=colors[1],
            alpha=0.5,
            label=f"psl_kde_{names[1]} > psl_kde_{names[0]}",
        )

        x_fill = np.linspace(0, 1, 1000)
        y_fill = np.minimum(psl_kde.evaluate(x_fill), psl_kde_2.evaluate(x_fill))

        ax.fill_between(x_fill, y_fill, color="purple", alpha=0.5)

        ax.plot(x, psl_kde.evaluate(x), color="b", label=f"PSL_KDE {names[0]}")
        ax.plot(x, psl_kde_2.evaluate(x), color="r", label=f"PSL_KDE {names[1]}")

        ax.legend()

        ax.set_title(f"PSL KDE Distributions Comparison \n P(PSL_KDE_{names[1]} > PSL_KDE_{names[0]}) $\\approx$ {p_greater}")

        return p_greater


class TeamPSL:
    """Class to Create the Bayesian Transition Matrix for a Team"""

    def __init__(self, EPL_Full_Data_Instance):
        self.EPL_Full = EPL_Full_Data_Instance

        self.epl = EPL_Full_Data_Instance.get_epl()
        self.epl_player_data = EPL_Full_Data_Instance.get_epl_player_data()
        self.R_storage = EPL_Full_Data_Instance.get_R_storage()
        self.Q_storage = EPL_Full_Data_Instance.get_Q_storage()
        self.player_kdes = EPL_Full_Data_Instance.get_player_kdes()
        self.partidos = EPL_Full_Data_Instance.get_partidos()

        self.team_id = None
        self.team_players_ids = []
        self.team_matches = []
        self.team_formations = []
        self.team_player_appearances = {}
        self.top_11_team_players = []

    def set_team(self, team_name):
        """Set the Team"""

        self.team_id = self.epl[self.epl["home_team_name"] == team_name][
            "home_team_id"
        ].values[0]
        self.team_matches = self.epl[self.epl["team_id"] == self.team_id][
            "match_id"
        ].unique()
        self.team_players_ids = (
            self.epl[self.epl["team_id"] == self.team_id]["player_id"].dropna().unique()
        )
        self.calculate_team_formations()
        self.calculate_top_11_players()

    def calculate_team_formations(self):
        """Calculate the Team Formations"""

        for pi, ti, li, lineup, _, team_id in self.EPL_Full:
            if team_id == self.team_id:
                duration = get_lineup_duration(lineup)
                players_ids = get_jugadores(lineup)
                self.team_formations.append((pi, ti, li, duration, set(players_ids)))

    def calculate_top_11_players(self):
        """Calculate the Top 11 Players

        Returns:
            List[int]: List of player IDs
        """

        self.calculate_team_player_appearances()

        self.top_11_team_players = sorted(
            self.team_player_appearances.items(),
            key=lambda x: x[1][1],
            reverse=True,
        )[:11]

        return [player_id for player_id, _ in self.top_11_team_players]

    def calculate_team_player_appearances(self):
        """Calculate the Team Player Appearances"""

        self.team_player_appearances = {}
        for _, _, _, duration, players_ids in self.team_formations:
            for player_id in players_ids:
                if player_id not in self.team_player_appearances:
                    self.team_player_appearances[player_id] = (0, 0)
                self.team_player_appearances[player_id] = (
                    self.team_player_appearances[player_id][0] + 1,
                    self.team_player_appearances[player_id][1] + duration,
                )

    def replace_player(self, old_player_name, new_player_name, list_players=None):
        """Replace a Player

        Args:
            old_player_name (str): Old Player Name, to be Replaced
            new_player_name (str): New Player Name, to be Replaced with

        Returns:
            List[int]: List of player IDs
        """
        if list_players is None:
            list_players = self.calculate_top_11_players()

        pos_old = None

        if old_player_name not in list_players:
            # Find a player in team with same position
            old_player_id = self.epl_player_data.get_player_id_by_name(old_player_name)
            old_player_pos = self.epl_player_data.get_player_position(old_player_id)
            print("Finding replacement for player with same position")

            print(f"Old Player: {old_player_name} - {old_player_id}")
            print(f"Old Player Position: {old_player_pos}")

            print(list_players)
            print(
                [
                    self.epl_player_data.get_player_position(player_id)
                    for player_id in list_players
                ]
            )

            for player_id in list_players:
                if (
                    self.epl_player_data.get_player_position(int(player_id)) == old_player_pos
                ):
                    pos_old = list_players.index(player_id)
                    break
        else:
            pos_old = list_players.index(
                self.epl_player_data.get_player_id_by_name(old_player_name)
            )

        if pos_old is None:
            raise ValueError("Unable to find a player to replace")

        list_players[pos_old] = self.epl_player_data.get_player_id_by_name(
            new_player_name
        )

        list_players = [
            float(player_id)
            for player_id in list_players
        ]

        return list_players

    def create_dist_TM(self, list_players):
        """Create the Distribution Transition Matrix"""

        matrix = np.zeros((14, 14))
        matrix = pd.DataFrame(
            matrix,
            columns=["G"] + list_players + ["L", "S"],
            index=["G"] + list_players + ["L", "S"],
        )

        dists_cache = {}

        for i, player_id in enumerate(list_players):
            matrix.loc["G", player_id] = self.player_kdes[player_id]["gains_prob"]
            matrix.loc[player_id, "L"] = self.player_kdes[player_id]["losses_prob"]
            matrix.loc[player_id, "S"] = self.player_kdes[player_id]["shots_prob"]

            for j, player_id_2 in enumerate(list_players):
                if i != j:
                    if (player_id, player_id_2) not in dists_cache:
                        dists_cache[(player_id, player_id_2)] = []

                    if (player_id_2, player_id) not in dists_cache:
                        dists_cache[(player_id_2, player_id)] = []

                    dists_cache[(player_id, player_id_2)].append(
                        self.player_kdes[player_id]["avg_pass_to_prob"]
                    )

                    dists_cache[(player_id_2, player_id)].append(
                        self.player_kdes[player_id_2]["avg_pass_from_prob"]
                    )

        return matrix, dists_cache

    def plot_dist_TM(self, matrix, dists_cache):
        """Plot the Distribution Transition Matrix

        Args:
            matrix (pd.DataFrame): Matrix with the distributions
            dists_cache (Dict[Tuple[int, int], List[gaussian_kde]]): Distributions Cache

            Call create_dist_TM to get the matrix and dists_cache
        """

        fig, axs_full = plt.subplots(15, 15, figsize=(16, 16))
        players = matrix.columns[1:-2]

        for ax in axs_full.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        axs_full[0, 0].axis("off")

        for i, player_id in enumerate(players):
            axs_full[0, i + 2].text(0.5, 0.5, int(player_id), ha="center", va="center")
            axs_full[i + 2, 0].text(0.5, 0.5, int(player_id), ha="center", va="center")

        axs_full[0, 1].text(0.5, 0.5, "G", ha="center", va="center")
        axs_full[1, 0].text(0.5, 0.5, "G", ha="center", va="center")

        axs_full[0, -2].text(0.5, 0.5, "L", ha="center", va="center")
        axs_full[0, -1].text(0.5, 0.5, "S", ha="center", va="center")

        axs_full[-2, 0].text(0.5, 0.5, "L", ha="center", va="center")
        axs_full[-1, 0].text(0.5, 0.5, "S", ha="center", va="center")

        for i in range(15):
            axs_full[i, i].axis("off")

            if i > 0:
                axs_full[-1, i].axis("off")
                axs_full[-2, i].axis("off")
                axs_full[i, 1].axis("off")
                axs_full[i, 1].text(0.5, 0.5, "0", ha="center", va="center")
                if i < 13:
                    axs_full[i, i].text(0.5, 0.5, "0", ha="center", va="center")
                    axs_full[-1, i].text(0.5, 0.5, "0", ha="center", va="center")
                    axs_full[-2, i].text(0.5, 0.5, "0", ha="center", va="center")
                else:
                    axs_full[i, i].text(0.5, 0.5, "1", ha="center", va="center")

        axs_full[1, -1].axis("off")
        axs_full[1, -2].axis("off")

        axs_full[1, -1].text(0.5, 0.5, "0", ha="center", va="center")
        axs_full[1, -2].text(0.5, 0.5, "0", ha="center", va="center")

        axs_full[-2, -1].text(0.5, 0.5, "0", ha="center", va="center")
        axs_full[-1, -2].text(0.5, 0.5, "0", ha="center", va="center")

        axs = axs_full[1:, 1:]

        for i, player_id in enumerate(players):
            axs[0, i + 1].plot(
                np.linspace(0, 1, 100),
                matrix.loc["G", player_id].evaluate(np.linspace(0, 1, 100)),
                color="blue",
            )

            axs[i + 1, -2].plot(
                np.linspace(0, 1, 100),
                matrix.loc[player_id, "L"].evaluate(np.linspace(0, 1, 100)),
                color="red",
            )

            axs[i + 1, -1].plot(
                np.linspace(0, 1, 100),
                matrix.loc[player_id, "S"].evaluate(np.linspace(0, 1, 100)),
                color="green",
            )

            for j, player_id_2 in enumerate(players):
                if i != j:
                    for dists in dists_cache[(player_id, player_id_2)]:
                        axs[i + 1, j + 1].plot(
                            np.linspace(0, 1, 100),
                            dists.evaluate(np.linspace(0, 1, 100)),
                            color="purple",
                            alpha=0.5,
                        )

                    for dists in dists_cache[(player_id_2, player_id)]:
                        axs[j + 1, i + 1].plot(
                            np.linspace(0, 1, 100),
                            dists.evaluate(np.linspace(0, 1, 100)),
                            color="teal",
                            alpha=0.5,
                        )

    def sample_R_from_dists(self, matrix, dists_cache, list_players):
        """Sample R from Distributions

        Args:
            matrix (pd.DataFrame): Matrix with the distributions
            dists_cache (Dict[Tuple[int, int], List[gaussian_kde]]): Distributions Cache
            list_players (List[int]): List of player IDs

        Returns:
            pd.DataFrame: Sampled R
        """

        R = np.zeros((14, 14))
        R = pd.DataFrame(
            R,
            columns=["G"] + list_players + ["L", "S"],
            index=["G"] + list_players + ["L", "S"],
        )

        def nonnegative(x):
            return x if x > 0 else 0

        for i, player_id in enumerate(list_players):
            R.loc["G", player_id] = nonnegative(
                matrix.loc["G", player_id].resample(1)[0][0]
                if isinstance(matrix.loc["G", player_id], gaussian_kde)
                else 0
            )
            R.loc[player_id, "L"] = nonnegative(
                matrix.loc[player_id, "L"].resample(1)[0][0]
                if isinstance(matrix.loc[player_id, "L"], gaussian_kde)
                else 0
            )
            R.loc[player_id, "S"] = nonnegative(
                matrix.loc[player_id, "S"].resample(1)[0][0]
                if isinstance(matrix.loc[player_id, "S"], gaussian_kde)
                else 0
            )

            for j, player_id_2 in enumerate(list_players):
                if i != j:
                    R.loc[player_id, player_id_2] = nonnegative(
                        np.mean(
                            [
                                ( 
                                    dist.resample(1)[0][0] 
                                    if isinstance(dist, gaussian_kde) 
                                    else 0
                                )
                                for dist in dists_cache[(player_id, player_id_2)]
                            ]
                        )
                    )
                    R.loc[player_id_2, player_id] = nonnegative(
                        np.mean(
                            [
                                (
                                    dist.resample(1)[0][0]
                                    if isinstance(dist, gaussian_kde)
                                    else 0
                                )
                                for dist in dists_cache[(player_id_2, player_id)]
                            ]
                        )
                    )

        return R

    def estimate_psl_distribution(self, psl_sample_counts, list_players):
        """Estimate the PSL Distribution

        Args:
            psl_sample_counts (int): Number of PSL samples
            list_players (List[int]): List of player IDs

        Returns:
            List[float]: List of PSL values
        """

        psls = []

        for _ in tqdm(range(psl_sample_counts)):
            R = self.sample_R_from_dists(
                *self.create_dist_TM(list_players), list_players
            ).values
            Q = build_Q(R)
            psls.append(psl_estimator(Q))

        return psls


class PlayerComparison:
    """Class to Compare Players impact on PSL Distribution"""

    def __init__(self, team_psl_instance: TeamPSL):
        self.team_psl = team_psl_instance

    def rank_players(
        self, base_player_name, compare_player_names, psl_sample_counts=1000
    ):
        """Rank Players by their impact on the PSL Distribution

        Args:
            base_player_name (str): Base Player Name
            compare_player_names (List[str]): List of Player Names to Compare
            psl_sample_counts (int, optional): Number of PSL samples. Defaults to 1000.

        Returns:
            List[Tuple[str, float]]: List of Player Names and their PSL Probability
        """

        base_player_id = self.team_psl.epl_player_data.get_player_id_by_name(
            base_player_name
        )
        list_players = self.team_psl.calculate_top_11_players()
        list_players = [
            int(player_id) for player_id in list_players
        ]

        if base_player_id not in list_players:

            # Si el jugador base no está en la lista de los 11 jugadores más utilizados
            # Replace the player with the player with the highest appearance and in the same position with the base player

            base_player_pos = self.team_psl.epl_player_data.get_player_position(base_player_id)
            for player_id in list_players:
                if self.team_psl.epl_player_data.get_player_position(player_id) == base_player_pos:
                    list_players[list_players.index(player_id)] = base_player_id
                    break

        pos_base = list_players.index(base_player_id)

        list_players = [float(player_id) for player_id in list_players]

        # Estimar la distribución PSL para el jugador base
        psls_base = self.team_psl.estimate_psl_distribution(
            psl_sample_counts, list_players
        )

        rankings = []

        for compare_player_name in compare_player_names:
            list_players_ = list_players.copy()
            list_players_[pos_base] = (
                self.team_psl.epl_player_data.get_player_id_by_name(compare_player_name)
            )
            psls_compare = self.team_psl.estimate_psl_distribution(
                psl_sample_counts, list_players_
            )
            p_greater = RandomVariablePSL.probability_psl_greater_than(
                psls_compare, psls_base
            )
            rankings.append((compare_player_name, p_greater))

        # Ordenar los jugadores por la probabilidad de mejorar la PSL
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


if __name__ == "__main__":
    EPL_Full_Data = EPL_Data(
        "../SampleData/epl.xlsx", "../SampleData/players.json", "R_storage.npy"
    )

    # epl = EPL_Full_Data.get_epl()
    # epl_player_data = EPL_Full_Data.get_epl_player_data()
    # R_storage = EPL_Full_Data.get_R_storage()
    # Q_storage = EPL_Full_Data.get_Q_storage()
    # partidos = EPL_Full_Data.get_partidos()
    # tp_ds = EPL_Full_Data.get_transition_prob_dataset()
    # player_ids = EPL_Full_Data.get_player_ids()
    # player_kdes = EPL_Full_Data.get_player_kdes()

    team_psl = TeamPSL(EPL_Full_Data)
    team_psl.set_team("Manchester City")

    player_comparison = PlayerComparison(team_psl)

    sorted_rankings = player_comparison.rank_players(
        "Sergio Agüero",
        [
            "Theo Walcott",
            "Franco Di Santo",
            "Olivier Giroud",
            "Wayne Rooney",
            "Romelu Lukaku",
            "Adam Le Fondre",
            "Jonathan Walters",
            "Marc-Antoine Fortuné",
            "Itay Shechter",
            "Grant Holt",
        ],
    )

    print(sorted_rankings)

    og_list = team_psl.calculate_top_11_players()

    RandomVariablePSL.plot_psl_distributions(
        team_psl.estimate_psl_distribution(1000, og_list),
        team_psl.estimate_psl_distribution(
            1000,
            team_psl.replace_player("Sergio Agüero", "Olivier Giroud"),
        ),
        "Sergio Agüero",
        "Olivier Giroud",
    )
