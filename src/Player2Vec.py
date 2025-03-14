""" Player2Vec using Node2Vec over Graph of Transition between player states """

# pylint: disable=E1126
# pylint: disable=E0401
# pylint: disable=C0413
# pylint: disable=C0411
# pylint: disable=C0103
# pylint: disable=C0301
# pylint: disable=W0612

import numpy as np
import pickle
import networkx as nx
import pandas as pd
from tqdm import tqdm

from src.bayesian_PSL import EPL_Data
from src.event_processing import separar_partido_en_equipo_pov
from src.futbol_types import TransitionMatrix
from src.match_data_extraction import get_lineup_duration
from gensim.models import Word2Vec
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# CLI Args
import argparse

def isPlayerId(x):
    if "_" in x:
        return False
    if "Loss" in x:
        return False
    if "Gain" in x:
        return False
    if "Shot" in x:
        return False
    try:
        int(x)
        return True
    except ValueError:
        # print(f"Player {player_id} not in model")
        return False
    return False


class Player2Vec:
    """ Class to train and use a Node2Vec model over the EPL player graph """

    def __init__(self, model_path=None, epl_data=None):
        self.model = None
        self.epl_data = epl_data
        self.epl_player_data = epl_data.get_epl_player_data()
        if model_path:
            self.load_model(model_path)

        self.dims = 0
        self.dim_red_embs = None

    def train(
        self,
        graph=None,
        dimensions=3,
        walk_length=16,
        num_walks=200,
        p=1,
        q=1,
        workers=4,
        window=12,
        min_count=1,
        batch_words=4,
    ):
        """ Train a Node2Vec model over the EPL player graph

        Args:
            graph (nx.Graph): Graph to train the model on
            dimensions (int): Dimension of the embeddings
            walk_length (int): Length of the random walks
            num_walks (int): Number of random walks
            workers (int): Number of workers
            window (int): Window size
            min_count (int): Minimum count
            batch_words (int): Batch words
        """

        if not graph:
            graph, _, _ = EPL_Graph(self.epl_data).build_graph()

        self.dims = dimensions
        node2vec = Node2Vec(
            graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            workers=workers,
            p=p,
            q=q,
        )
        self.model = node2vec.fit(
            window=window, min_count=min_count, batch_words=batch_words
        )

        self.dim_red_embs = self.dim_reduce(n_components=3)

    def load_model(self, model_path):
        """ Load a Node2Vec model from a file

        Args:
            model_path (str): Path to the model file
        """

        self.model = Word2Vec.load(model_path)

    def save_model(self, model_path):
        """ Save the Node2Vec model to a file

        Args:
            model_path (str): Path to save the model
        """
        self.model.save(model_path)

    def get_ids(self):
        """ Get the node IDs from the model

        Returns:
            List[int]: List of node ids
        """

        return self.model.wv.index_to_key

    def get_similar_players(self, player_name, topn=10, same_position=True):
        """ Get the most similar players to a player

        Args:

            player_name (str): Name of the player
            topn (int): Number of similar players to return
            same_position (bool): Whether to return players with the same position

        Returns:
            List[Tuple[str, int, str, float, str]]: List of similar players
        """
        player_id = self.get_player_id(player_name)
        sample_team = self.get_team_name(player_id)
        sample_position = self.get_player_position(player_id)

        total = len([x for x in self.get_ids() if "___" in x])

        most_similar = self.model.wv.most_similar(f"{player_id}___", topn=total)
        count = 0
        results = []
        for player_id, similarity in most_similar:
            if count == topn:
                break
            if player_id.split("_")[0] in ["Gain", "Loss", "Shot"]:
                continue
            # if "_" in player_id and "___" not in player_id:
            # continue

            player_id = self.rework_id(player_id)
            player_name = self.epl_player_data.get_player_name(player_id)
            team = self.get_team_name(player_id)
            position = self.get_player_position(player_id)

            if sample_team != team and (
                position == sample_position or not same_position
            ):
                count += 1
                results.append((player_name, player_id, team, similarity, position))
        return results

    def get_player_id(self, player_name):
        """ Get the player ID from the player name

        Args:
            player_name (str): Name of the player

        Returns:
            int: Player ID
        """

        return self.epl_player_data.get_player_id_by_name(player_name)

    def get_team_name(self, player_id):
        """ Get the team name of a player

        Args:
            player_id (int): Player ID

        Returns:
            str: Team name
        """

        for pi, partido in enumerate(self.epl_data.partidos):
            for ti, equipo in enumerate(
                separar_partido_en_equipo_pov(partido)
            ):
                if player_id in equipo["player_id"].unique():
                    return partido["home_team_name"].iloc[0]
        return None

    def get_player_position(self, player_id):
        """ Get the player position from the player ID

        Args:
            player_id (int): Player ID

        Returns:
            str: Player position
        """

        return self.epl_player_data.get_player_position(player_id)

    def rework_id(self, x):
        """ Rework the ID to get the player ID

        Args:
            x (str): ID

        Returns:
            int: Player ID
        """

        return int(x.split("_")[0])

    def get_embedding(self, player_id):
        """ Get the embedding of a player

        Args:
            player_id (int): Player ID

        Returns:
            np.array: Player embedding
        """

        id_ = player_id if "___" in str(player_id) else f"{player_id}___"

        if id_ in self.model.wv:
            return self.model.wv[id_]

        return None

    def dim_reduce(self, n_components=3, method="PCA"):
        """ Reduce the dimensions of the player embedding

        Args:
            player_id (int): Player ID
            n_components (int): Number of components

        Returns:
            np.array: Reduced dimensions
        """

        # Using TSNE, reduce the dimensions of all the embeddings

        ids = self.get_ids()
        # ids = list(filter(lambda x: "_" not in x, ids))
        # emb = np.array([self.get_embedding(int(x)) for x in ids])
        emb = np.array([self.model.wv[x] for x in ids])

        dim_reducer = PCA(n_components=n_components) if method == "PCA" else TSNE(n_components=n_components)
        dim_red_embs = dim_reducer.fit_transform(emb)
        self.dim_red_embs = {
            ids[i]: dim_red_embs[i]
            for i in range(len(ids))
            if isPlayerId(ids[i])
        }
        return self.dim_red_embs

    def get_reduced_embeddings(self, player_id):
        return self.dim_red_embs[player_id]

    def export_embeddings_json(self, output_path, emb_dims=3):

        if self.dims != emb_dims and self.dim_red_embs is None:
            self.dim_red_embs = self.dim_reduce(n_components=emb_dims)

        shots_prob_emb_dict = {}
        ids = self.get_ids()
        ids = list(filter(lambda x: "_" not in x, ids))
        iterator = tqdm(ids, desc="Exporting Embeddings", total=len(ids))
        for player_id in iterator:
            if str(player_id) not in self.model.wv.index_to_key:
                # print(f"Player {player_id} not in model")
                continue
            # if player_id is not int castable
            try:
                int(player_id)
            except ValueError:
                # print(f"Player {player_id} not in model")
                continue

            # print("Player ID", player_id)

            iterator.set_postfix_str(f"Player {player_id}")

            # emb = self.model.wv.get_vector(str(player_id))
            if self.dims == emb_dims:
                emb = self.get_embedding(player_id)
            else:
                emb = self.get_reduced_embeddings(player_id)

            # kde = player_kdes_df.loc[player_id, "shots_prob"]
            # if kde != 0:
            p_data = {
                **{f"emb_{i}": emb[i] for i in range(len(emb))},
                # **{f"shots_prob_{i}": kde.pdf(x) for i, x in enumerate(x_space)},
                # "shots_prob": kde,
                "name": self.epl_data.epl_player_data.get_player_name(int(player_id)),
                "position": self.epl_data.epl_player_data.get_player_position(int(player_id)),
                "team": self.epl_data.get_player_team(int(player_id)),
                "id": player_id,
            }

            shots_prob_emb_dict[player_id] = p_data

        shots_prob_emb_ds = pd.DataFrame(shots_prob_emb_dict).T

        shots_prob_emb_ds.T.to_json(output_path)

    def __repr__(self):
        return f"Player2Vec(model={self.model}, epl_data={self.epl_data.__repr__()}, dims={self.dims})"


class EPL_Graph:
    """ Class to build the EPL player graph """

    def __init__(self, epl_data: EPL_Data):
        self.epl_data = epl_data
        self.epl_player_data = epl_data.get_epl_player_data()
        self.Q_storage = epl_data.get_Q_storage()
        self.R_storage = epl_data.get_R_storage()
        self.partidos = epl_data.get_partidos()

        self.player_total_duration = 0

        self.graph = nx.Graph()
        self.sub_graphs = {}
        self.players_sub_nodes = {}

        self.durations = {}
        self.total_duration = 0

    def R_to_Graph(self, R: TransitionMatrix, names: list[str]):
        """
        Convert a Transition Matrix to a Directed Graph

        Args:
            R (TransitionMatrix): Transition Matrix
            names (list[str]): List of player names

        Returns:
            nx.DiGraph: Directed Graph
        """

        G = nx.DiGraph()

        for i, name in enumerate(names):
            player_name = self.epl_player_data.get_player_name(name)
            player_name = player_name if player_name else name

            # Shot Ratio is Q[1+i, 14]
            shot_ratio = R[1 + i, 14]
            G.add_node(
                player_name,
                position=self.epl_player_data.get_player_position(name),
                shot_ratio=shot_ratio
            )

        for i, name in enumerate(names):
            player_name = self.epl_player_data.get_player_name(name)
            player_name = player_name if player_name else name

            for j, name2 in enumerate(names):
                player_name2 = self.epl_player_data.get_player_name(name2)
                player_name2 = player_name2 if player_name2 else name2
                if i == j:
                    continue
                if R[1 + i, 1 + j] > 0:
                    G.add_edge(player_name, player_name2, weight=R[1 + i, 1 + j])

        return G

    def R_to_full_graph(self, R: TransitionMatrix, names: list[str], ti="", pi="", li="", use_weighted_edges=False):
        """
        Convert a Transition Matrix to a Directed Graph

        Args:
            R (TransitionMatrix): Transition Matrix
            names (list[str]): List of player names
            ti (str): Team index
            pi (str): Player index
            li (str): Lineup index

        Returns:
            nx.DiGraph: Directed Graph
        """

        G = nx.DiGraph()

        suffix = f"_{ti}_{pi}_{li}" if (ti != "" and pi != "" and li != "") else ""

        gain_state = "Gain" + suffix
        loss_state = "Loss" + suffix
        shot_state = "Shot" + suffix

        G.add_node(gain_state, position="Gain")
        G.add_node(loss_state, position="Loss")
        G.add_node(shot_state, position="Shot")

        for i, name in enumerate(names):
            player_i_state = f"{name}_{ti}_{pi}_{li}"

            # Shot Ratio is Q[1+i, 14]
            shot_ratio = R[2 + i, 14]
            G.add_node(
                player_i_state,
                position=self.epl_player_data.get_player_position(name),
                shot_ratio=shot_ratio
            )

            # Add edge to the Gain state, the weight is the probability of the player gaining possession
            gain_w = R[1, 2 + i]
            if gain_w < 0 or not np.isfinite(gain_w):
                gain_w = 0

            # Add edge to the Loss state, the weight is the probability of the player losing possession
            loss_w = R[2 + i, 13]
            if loss_w < 0 or not np.isfinite(loss_w):
                loss_w = 0

            # Add edge to the Shot state, the weight is the probability of the player taking a shot
            shot_w = R[2 + i, 14]
            if shot_w < 0 or not np.isfinite(shot_w):
                shot_w = 0

            if use_weighted_edges:
                G.add_edge(gain_state, player_i_state, weight=gain_w)
                G.add_edge(player_i_state, loss_state, weight=loss_w)
                G.add_edge(player_i_state, shot_state, weight=shot_w)
            else:
                G.add_edge(gain_state, player_i_state)
                G.add_edge(player_i_state, loss_state)
                G.add_edge(player_i_state, shot_state)

        for i, name in enumerate(names):
            player_i_state = f"{name}_{ti}_{pi}_{li}"
            # player_name = epl_player_data.get_player_name(name)
            # player_name = player_name if player_name else name

            for j, name2 in enumerate(names):
                # player_name2 = epl_player_data.get_player_name(name2)
                # player_name2 = player_name2 if player_name2 else name2
                if i == j:
                    continue
                player_j_state = f"{name2}_{ti}_{pi}_{li}"

                w = R[2 + i, 2 + j]

                if w < 0 or not np.isfinite(w):
                    w = 0

                if use_weighted_edges:
                    G.add_edge(
                        player_i_state,
                        player_j_state,
                        weight=w
                    )
                else:
                    G.add_edge(
                        player_i_state,
                        player_j_state
                    )

        return G

    def precalculate_durations(self):
        """ Calculate the total duration of each player in the EPL """
        self.durations = {}
        self.total_duration = 0

        self.player_total_duration = {}
        for pi, ti, li, lineup, match_id, team_id in self.epl_data:
            R = self.R_storage[pi, ti, li]
            ids = [int(x) for x in list(R[1:, 0][1:-2])]
            duration = get_lineup_duration(lineup)
            self.durations[(pi, ti, li)] = duration
            self.total_duration += duration
            for id_ in ids:
                if id_ not in self.player_total_duration:
                    self.player_total_duration[id_] = 0
                self.player_total_duration[id_] += duration

    def build_graph(self, use_Q=False, weight_player_to_state=True, n_matches=None):
        """ Build the EPL player graph

        Args:
            use_Q (bool): Whether to use the Q matrix (Normalized Transition Matrix)
            weight_player_to_state (bool): Whether to weight the player to initial/final state edges

        Returns:
            nx.DiGraph: Directed Graph
            List[nx.DiGraph]: List of Directed Graphs for each lineup
            Dict[int, List[str]]: Dictionary of player subnodes
        """

        self.precalculate_durations()

        graph = nx.DiGraph()

        # Add nodes for all the players
        for player_id in self.epl_data.get_player_ids():
            if player_id == 0:
                continue
            graph.add_node(
                int(player_id),
                position=self.epl_player_data.get_player_position(player_id),
                duration=self.epl_data.get_player_total_duration(float(player_id)),
            )

        # Add nodes for special states
        graph.add_node("Gain", position="Gain")
        graph.add_node("Loss", position="Loss")
        graph.add_node("Shot", position="Shot")

        # Big Graph with all the players for all the teams for all the matches in the EPL

        graphs = {}
        players_sub_nodes = {}

        for pi, ti, li, lineup, _, _ in tqdm(self.epl_data, desc="Building Graph", total=len(self.epl_data)):
            if n_matches and pi >= n_matches:
                break
            Q = self.Q_storage[pi, ti, li]
            R = self.R_storage[pi, ti, li]
            duration = self.durations[(pi, ti, li)]
            names = [int(x) for x in list(R[1:, 0][1:-2])]

            # G = R_to_full_graph(R, names)
            if use_Q:
                G = self.R_to_full_graph(Q, names, ti, pi, li, use_weighted_edges=weight_player_to_state)
            else:
                G = self.R_to_full_graph(R, names, ti, pi, li, use_weighted_edges=weight_player_to_state)

            graphs[(pi, ti, li)] = G

            graph = nx.compose(graph, G)

            for name in names:
                if name == 0:
                    continue
                # if name is not int castable
                if not isinstance(name, int):
                    continue

                player_i_state = f"{name}_{ti}_{pi}_{li}"
                if name not in players_sub_nodes:
                    players_sub_nodes[name] = []
                players_sub_nodes[name].append(player_i_state)

                # Add node from name to player_i_state

                if weight_player_to_state:
                    if self.player_total_duration[name] > 0:
                        weight = duration / self.player_total_duration[name]
                    else:
                        weight = 0

                    if not np.isfinite(weight):
                        weight = 0

                    graph.add_edge(name, player_i_state, weight=weight)
                else:
                    graph.add_edge(name, player_i_state)

            # Add edge from Gain to Gain_{ti}_{pi}_{li}
            if weight_player_to_state:
                graph.add_edge("Gain", f"Gain_{ti}_{pi}_{li}", weight=1)
                graph.add_edge(f"Loss_{ti}_{pi}_{li}", "Loss", weight=1)
                graph.add_edge(f"Shot_{ti}_{pi}_{li}", "Shot", weight=1)
            else:
                graph.add_edge("Gain", f"Gain_{ti}_{pi}_{li}")
                graph.add_edge(f"Loss_{ti}_{pi}_{li}", "Loss")
                graph.add_edge(f"Shot_{ti}_{pi}_{li}", "Shot")

            # if any existing edge is infinite print the nodes
            for u, v, d in G.edges(data=True):
                if not np.isfinite(d["weight"]):
                    print(u, v, d)

        self.graph = graph
        self.sub_graphs = graphs
        self.players_sub_nodes = players_sub_nodes
        return graph, graphs, players_sub_nodes

    def get_graph(self):
        """ Get the EPL player graph

        Returns:
            nx.DiGraph: Directed Graph
        """

        return self.graph

    def save_graph(self, path):
        """ Save the EPL player graph to a file

        Args:
            path (str): Path to save the graph
        """

        with open(path, "wb") as f:
            pickle.dump(self.graph, f)

    def load_graph(self, path):

        """ Load the EPL player graph from a file

        Args:
            path (str): Path to load the graph
        """

        with open(path, "rb") as f:
            self.graph = pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Node2Vec model over the EPL player graph")
    parser.add_argument(
        "--model_path", type=str, help="Path to the model file", default=None
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to save the model", default=None
    )
    parser.add_argument(
        "--dimensions", type=int, help="Dimension of the embeddings", default=3
    )
    parser.add_argument(
        "--walk_length", type=int, help="Length of the random walks", default=16
    )
    parser.add_argument(
        "--num_walks", type=int, help="Number of random walks", default=200
    )
    parser.add_argument("--p", type=int, help="p", default=1)
    parser.add_argument("--q", type=int, help="q", default=1)
    parser.add_argument("--workers", type=int, help="Number of workers", default=4)
    parser.add_argument("--window", type=int, help="Window size", default=12)
    parser.add_argument("--min_count", type=int, help="Minimum count", default=1)
    parser.add_argument("--batch_words", type=int, help="Batch words", default=4)

    # "../SampleData/epl.xlsx", "../SampleData/players.json", "R_storage.npy"
    parser.add_argument(
        "--epl_data_path",
        type=str,
        help="Path to the EPL data",
        default="../SampleData/epl.xlsx",
    )

    parser.add_argument(
        "--players_path",
        type=str,
        help="Path to the players data",
        default="../SampleData/players.json",
    )

    parser.add_argument(
        "--r_storage_path",
        type=str,
        help="Path to the R storage",
        default="R_storage.npy",
    )

    args = parser.parse_args()

    epl_data = EPL_Data(args.epl_data_path, args.players_path, args.r_storage_path)
    player2vec = Player2Vec(model_path=args.model_path, epl_data=epl_data)
    graph, _, _ = EPL_Graph(epl_data).build_graph()
    player2vec.train(
        graph=graph,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        p=args.p,
        q=args.q,
        workers=args.workers,
        window=args.window,
        min_count=args.min_count,
        batch_words=args.batch_words,
    )

    player2vec.save_model(args.output_path)
    player2vec.export_embeddings_json(args.output_path.replace(".model", ".json"))
