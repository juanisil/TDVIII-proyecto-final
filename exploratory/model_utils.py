import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression


from IPython.display import display, HTML

import itertools


import sys

sys.path.append("../")

from src.event_processing import leer_excel
from src.event_processing import (
    separar_partido_del_equipo_en_lineups,
    separar_partido_en_equipo_pov,
    separar_partidos,
)
from src.match_data_extraction import get_jugadores, get_lineup_duration
from src.utils_CTMC import psl_estimator, build_Q, build_R
from src.utils_CTMC import get_ratio_gains, get_ratio_loss, get_ratio_shots

from src.futbol_types import TransitionMatrix
from typing import List

from sklearn.base import RegressorMixin


# ## Dataset

ds = pd.read_csv("dataset.csv")

ds = ds[ds["target"] != 0]

ds["date"] = pd.to_datetime(ds["date"])
dates = ds["date"]
date_2_3 = dates.mean() + 2 / 3 * dates.std()

# For a sample player_1 in ds, plot target distribution
players_ocurrences = ds["player_1"].value_counts().sort_values(ascending=False)
sample_player = players_ocurrences.index[0]

sample_player_ds = ds[ds["player_1"] == sample_player]
sample_player_ds["target"].hist(bins=100)

left_c = ds[ds["date"] < date_2_3].shape[0]
right_c = ds[ds["date"] >= date_2_3].shape[0]

# ### Train-Test Split

# Select pairs of player_1 and player_2 that appear in the dataset
# Keep them in a test set

pairs = ds[["player_1", "player_2"]].drop_duplicates()
sample_pairs = pairs.sample(100)

test = ds[
    ds["player_1"].isin(sample_pairs["player_1"])
    & ds["player_2"].isin(sample_pairs["player_2"])
]
train = ds[
    ~ds["player_1"].isin(sample_pairs["player_1"])
    | ~ds["player_2"].isin(sample_pairs["player_2"])
    | ~ds["player_1"].isin(sample_pairs["player_2"])
    | ~ds["player_2"].isin(sample_pairs["player_1"])
]

train_left = train[train["date"] < date_2_3]
test_right = test[test["date"] >= date_2_3]


X_train = train_left.drop(
    columns=["player_1", "player_2", "target", "date", "partido_id"]
)
y_train = train_left["target"]

X_test = test_right.drop(
    columns=["player_1", "player_2", "target", "date", "partido_id"]
)
y_test = test_right["target"]
y_test = np.array(y_test)


# Player Custom Features

epl = leer_excel("../SampleData/epl.xlsx")

partidos = separar_partidos(epl)
partidos = [partido for partido in partidos if partido["date"].values[0] > date_2_3]

sample_partido = partidos[np.random.randint(len(partidos))]

equipos = separar_partido_en_equipo_pov(sample_partido)
sample_equipo = equipos[np.random.randint(2)]
lineups = separar_partido_del_equipo_en_lineups(sample_equipo)
sample_lineup = lineups[int(np.random.randint(2))]
jugadores = get_jugadores(sample_lineup)

jugador = jugadores[np.random.randint(len(jugadores))]
lineups = separar_partido_del_equipo_en_lineups(sample_equipo)
psls_diffs = np.array([psl_estimator(build_Q(build_R(lineup))) for lineup in lineups])
lineup_durations = np.array([get_lineup_duration(lineup) for lineup in lineups])


def get_features(player_1, partido):
    # For a given player, get aggregated features for all matches played before partido
    # home_team_id if player_1 is in home team, away_team_id if player_1 is in away team
    team_id = partido[partido["player_id"] == player_1]["team_id"].values[0]

    player_data = {
        "pases/90": 0,
        "shots/90": 0,
        "losses/90": 0,
        "gains/90": 0,
        "minutes": 0,
    }

    # Get all matches played
    date = partido["date"].values[0]
    prev_matches = epl[epl["date"] < date]
    for partido in separar_partidos(prev_matches):
        for equipo in separar_partido_en_equipo_pov(partido):
            if equipo["team_id"].values[0] == team_id:
                if player_1 in get_jugadores(equipo):
                    for lineup in separar_partido_del_equipo_en_lineups(equipo):
                        players = get_jugadores(lineup)
                        if player_1 in players:
                            minutes = get_lineup_duration(lineup)
                            if minutes == 0:
                                continue

                            player_data["shots/90"] += get_ratio_shots(lineup, player_1)
                            player_data["losses/90"] += get_ratio_loss(lineup, player_1)
                            player_data["gains/90"] += get_ratio_gains(lineup, player_1)

                            player_data["minutes"] += minutes

                            passes = lineup[
                                (lineup["player_id"] == player_1)
                                & (lineup["type"] == 1)
                                & (lineup["outcome"] == 1)
                            ]

                            player_data["pases/90"] += passes.shape[0] / minutes

    return player_data


def predicted_psl(
    p1: int, lineups: List[List[int]], model: RegressorMixin
) -> TransitionMatrix:
    # Given Q and p1, update value for passes from p1 to every other player, and from every other player to p1
    counter = 0
    psls = []
    for lineup in lineups:
        jugadores = get_jugadores(lineup)
        R = build_R(lineup)
        for i, player in enumerate(jugadores):
            if player == p1:
                p1_features = pd.DataFrame([get_features(player, lineup)])
                for j, player2 in enumerate(jugadores):
                    if player2 != player:
                        counter += 1
                        p2_features = pd.DataFrame([get_features(player2, lineup)])
                        R[i + 1, j + 1] = model.predict(
                            pd.concat([p1_features, p2_features]).values.reshape(1, -1)
                        )[0]
                        R[j + 1, i + 1] = model.predict(
                            pd.concat([p2_features, p1_features]).values.reshape(1, -1)
                        )[0]

        Q = build_Q(R)
        psls.append(psl_estimator(Q))

    # print(counter)
    lineup_durations = np.array([get_lineup_duration(lineup) for lineup in lineups])

    return np.average(psls, weights=lineup_durations)


# Precomputing

max_n_lineups = np.max(
    [
        len(separar_partido_del_equipo_en_lineups(equipo))
        for partido in separar_partidos(epl)
        for equipo in separar_partido_en_equipo_pov(partido)
    ]
)

# Precompute R

n_matches = len(separar_partidos(epl))

R_storage = np.zeros((n_matches, 2, max_n_lineups, 15, 15))


def precompute_R_matrices(matches, R_storage):
    for pi, partido in enumerate(tqdm(separar_partidos(matches))):
        try:
            match_id = partido["match_id"].values[0]
        except:
            continue

        for team_index, equipo in enumerate(separar_partido_en_equipo_pov(partido)):
            lineups = separar_partido_del_equipo_en_lineups(equipo)

            for lineup_index, lineup in enumerate(lineups):
                jugadores = get_jugadores(lineup)

                # Build the R matrix (14x14)
                R = build_R(lineup)

                # Indexes 1 through 11 are players, both columns and rows
                # Store the R matrix in the storage last 14x14 part of the 15x15 matrix,
                # Save the lineup index in the first column from 1 to 11
                # Also in the first row
                if len(jugadores) > 11:
                    continue

                jugadores = np.array(jugadores)
                if len(jugadores) < 11:
                    jugadores = np.pad(jugadores, (0, 11 - len(jugadores)))
                R_storage[pi, team_index, lineup_index, 1:, 1:] = R
                R_storage[pi, team_index, lineup_index, 0, 2:13] = jugadores
                R_storage[pi, team_index, lineup_index, 2:13, 0] = jugadores


precompute_R_matrices(epl, R_storage)

# Save R_storage
np.save("R_storage.npy", R_storage)

R_storage = np.load("R_storage.npy")


def visualize_R_matrix(R):
    # 0, 1:12 are player ids,
    # 1:12, 0 are also player ids,
    # 1:, 1: is the true R matrix, :2f precision

    html = "<table>"
    for i in range(15):
        html += "<tr>"
        for j in range(15):
            if i == 0 and j == 0:
                html += "<td></td>"  # Empty top-left cell
            elif i == 0 and j == 1 or i == 1 and j == 0:  # Gain state col/row headers
                html += "<td><b>G</b></td>"
            elif i == 0 and j == 13 or i == 13 and j == 0:  # Loss state col/row headers
                html += "<td><b>L</b></td>"
            elif (
                i == 0 and j == 14 or i == 14 and j == 0
            ):  # Shots state col/row headers
                html += "<td><b>S</b></td>"
            elif i == 0:
                html += f"<td><b>{int(R[0, j])}</b></td>"  # Column headers
            elif j == 0:
                html += f"<td><b>{int(R[i, 0])}</b></td>"  # Row headers
            else:
                html += f"<td>{R[i, j]:.2f}</td>"  # Data cells
        html += "</tr>"
    html += "</table>"

    display(HTML(html))


visualize_R_matrix(R_storage[0, 0, 0])


def R_to_DataFrame(R, jugadores):
    # Given a 14x14 R matrix, and a list of jugadores,
    # return a DataFrame with the R matrix in 1:, 1:
    # and the jugadores in the first row and column, also a G, L, S index/column

    cols = ["G"] + list(jugadores) + ["L", "S"]
    df = pd.DataFrame(R[1:, 1:], index=cols, columns=cols)
    # df.index.name = "State"
    df.columns.name = "State"

    return df


# R_storage is of shape (n_matches=380, 2, max_n_lineups=4, 15, 15)
# Map the R_to_DataFrame to the last two dimensions of R_storage for every other dim
# But Timing it


# Precompute Features


def precompute_and_store_features_optimized(epl):
    # Instead of going through all the previous matches for each player in each new match,
    # we can use the precomputed features to get the data for each player in each match

    player_stats = []

    all_matches = separar_partidos(epl)

    # Solving with Dinamic Programming
    # For each match m,
    # the features of a player in the match m
    # are the stats of the player until match m-1 included
    # stats are: pases/90, shots/90, losses/90, gains/90, minutes

    # The features of a player in the match 0 are all stats = 0

    # First get player_stats for all players in all matches
    # Then get the features for each player in each match

    for partido in tqdm(all_matches):
        match_id = partido["match_id"].values[0]
        date = partido["date"].values[0]

        for equipo in separar_partido_en_equipo_pov(partido):
            team_id = equipo["team_id"].values[0]
            jugadores = get_jugadores(equipo)

            for jugador in jugadores:
                player_data = {
                    "player_id": jugador,
                    "match_id": match_id,
                    "pases/90": 0,
                    "shots/90": 0,
                    "losses/90": 0,
                    "gains/90": 0,
                    "minutes": 0,
                    "match_date": date,
                }

                for lineup in separar_partido_del_equipo_en_lineups(equipo):
                    if jugador in get_jugadores(lineup):
                        minutes = get_lineup_duration(lineup)
                        if minutes == 0:
                            continue

                        player_data["shots/90"] += get_ratio_shots(lineup, jugador)
                        player_data["losses/90"] += get_ratio_loss(lineup, jugador)
                        player_data["gains/90"] += get_ratio_gains(lineup, jugador)
                        player_data["minutes"] += minutes

                        pases = lineup[
                            (lineup["player_id"] == jugador)
                            & (lineup["type"] == 1)
                            & (lineup["outcome"] == 1)
                        ]
                        player_data["pases/90"] += pases.shape[0] / minutes

                player_stats.append(player_data)

    # Create DataFrame with MultiIndex
    player_stats_df = pd.DataFrame(player_stats)
    player_stats_df["match_date"] = pd.to_datetime(player_stats_df["match_date"])

    # Second part

    # Sort all_matches by date, newest first
    all_matches = sorted(all_matches, key=lambda x: x["date"].values[0], reverse=True)

    player_features_df = pd.DataFrame(
        columns=[
            "player_id",
            "match_id",
            "pases/90",
            "shots/90",
            "losses/90",
            "gains/90",
            "minutes",
            "match_date",
        ],
    )

    # For each match, get the features for each player
    for partido in tqdm(all_matches):
        match_id = partido["match_id"].values[0]
        date = partido["date"].values[0]

        for equipo in separar_partido_en_equipo_pov(partido):
            team_id = equipo["team_id"].values[0]
            jugadores = get_jugadores(equipo)

            for jugador in jugadores:
                # Get the stats for the player until match m-1 included (match_id < match_id)
                # Sum all the stats for the player in all matches until match m-1 included

                prev_stats = player_stats_df[
                    (player_stats_df["match_date"] < date)
                    & (player_stats_df["player_id"] == jugador)
                ]
                # TypeError: 'DatetimeArray' with dtype datetime64[ns] does not support reduction 'sum'
                # Drop the match_date column
                prev_stats = prev_stats.drop(
                    columns=["match_date", "match_id", "player_id"]
                )
                prev_stats_sum = prev_stats.sum()

                player_data = {
                    "player_id": jugador,
                    "match_id": match_id,
                    "pases/90": prev_stats_sum["pases/90"],
                    "shots/90": prev_stats_sum["shots/90"],
                    "losses/90": prev_stats_sum["losses/90"],
                    "gains/90": prev_stats_sum["gains/90"],
                    "minutes": prev_stats_sum["minutes"],
                    "match_date": date,
                }

                player_data_df = pd.DataFrame([player_data])
                player_features_df = pd.concat(
                    [player_features_df, player_data_df], ignore_index=True
                )

    return player_features_df.set_index(["player_id", "match_id"])


# ## Load Player Features


def load_player_features_dataframe(filepath):
    # Load the DataFrame from disk
    return pd.read_parquet(filepath)


def get_features_from_df(player_id, match_id, features_df, cols):
    try:
        return features_df.loc[(player_id, match_id)][cols].to_dict()

    except KeyError:
        return {
            "pases/90": 0,
            "shots/90": 0,
            "losses/90": 0,
            "gains/90": 0,
            "minutes": 0,
        }


# PSL Tools

def replace_q_values(
    R_df,
    model,
    p1,
    p2,
    features_df,
    p1_features=None,
    model_cols=list(set([col[:-2] for col in X_train.columns])),
):
    # Given a 14x14 R matrix, replace the q values for p1, p2 with the model predictions
    # Get the features for p1 and p2 from the features_df
    # Get the model predictions for p1, p2
    # Replace the values in the R matrix
    # Return the updated R matrix

    if p1_features is None:
        p1_features = pd.DataFrame(
            [get_features_from_df(p1, 0, features_df, model_cols)]
        )

    p2_features = pd.DataFrame([get_features_from_df(p2, 0, features_df, model_cols)])

    p1_features = p1_features[model_cols]
    p2_features = p2_features[model_cols]

    q_1_2 = model.predict(pd.concat([p1_features, p2_features]).values.reshape(1, -1))[
        0
    ]

    q_2_1 = model.predict(pd.concat([p2_features, p1_features]).values.reshape(1, -1))[
        0
    ]

    R_df.loc[p1, p2] = q_1_2
    R_df.loc[p2, p1] = q_2_1

    return R_df, q_1_2, q_2_1


match_id_RS_index_map = {
    m: i
    for i, m in enumerate(
        [partido["match_id"].values[0] for partido in separar_partidos(epl)]
    )
}


def match_id_2_RS_index(match_id):
    return match_id_RS_index_map[match_id]


def get_psl_diffs(
    R_storage,
    model,
    sample_pairs,
    matches,
    player_features_df,
    model_cols=list(set([col[:-2] for col in X_train.columns])),
):
    psls_diffs = []
    partidos = separar_partidos(matches)
    # pi_overhead = 380 - len(partidos)
    no_updates = 0
    updates = 0

    iterator = tqdm(partidos)
    for pi, partido in enumerate(iterator):
        try:
            match_id = partido["match_id"].values[0]
        except:
            continue

        for ti, equipo in enumerate(separar_partido_en_equipo_pov(partido)):
            team_psls = []
            team_alt_psls = []

            lineups = separar_partido_del_equipo_en_lineups(equipo)
            lineup_durations = np.array(
                [get_lineup_duration(lineup) for lineup in lineups]
            )

            for li, lineup in enumerate(lineups):
                jugadores = get_jugadores(lineup)
                team_rs_index = match_id_2_RS_index(match_id)
                R_f_s = R_storage[team_rs_index, ti, li].copy()
                # print("R from Storage:", R_f_s.shape)
                R_df = R_to_DataFrame(R_f_s, R_f_s[0, 2:13])
                # print("R DF:", R_df.shape)

                R = R_f_s[1:, 1:]
                # print("R 14x14: ", R.shape)
                real_psl = psl_estimator(build_Q(R))
                team_psls.append(real_psl)

                R_mod = R.copy()

                # Check jugadores are in R_df [0, 2:13]
                if not all([jugador in R_df.columns for jugador in jugadores]):
                    # print(pi + pi_overhead, ti, li, jugadores)
                    # print(lineup_durations[li])
                    # visualize_R_matrix(R_f_s)
                    team_alt_psls.append(0)
                    continue

                for i, player in enumerate(jugadores):
                    if player in sample_pairs["player_1"].values:
                        p1_features = pd.DataFrame(
                            [
                                get_features_from_df(
                                    player, match_id, player_features_df, model_cols
                                )
                            ]
                        )

                        for j, player2 in enumerate(jugadores):
                            if player2 != player:
                                # tmp = R_df.loc[player, player2]
                                tmp_R = R[i + 1, j + 1]
                                R_df, q_1_2, q_2_1 = replace_q_values(
                                    R_df,
                                    model,
                                    player,
                                    player2,
                                    player_features_df,
                                    p1_features,
                                    model_cols,
                                )
                                R_mod[i + 1, j + 1] = q_1_2
                                R_mod[j + 1, i + 1] = q_2_1
                                # post = R_df.loc[player, player2]
                                post_R = R_mod[i + 1, j + 1]
                                # if tmp == post:
                                # raise ValueError("No update")

                                if tmp_R == post_R:
                                    no_updates += 1
                                else:
                                    updates += 1
                                    # print(no_updates)
                                    # raise ValueError("No update")

                # R = np.array(R_df.values)
                # print(R.shape)
                modified_psl = psl_estimator(build_Q(R_mod))
                if real_psl == modified_psl:
                    iterator.set_postfix(
                        {
                            "no_updates": no_updates,
                            "updates": updates,
                        }
                    )

                    # raise ValueError("No update")
                team_alt_psls.append(modified_psl)

            psl = np.average(team_psls, weights=lineup_durations)
            psl_alt = np.average(team_alt_psls, weights=lineup_durations)
            psls_diffs.append((psl, psl_alt))

    return psls_diffs


def evaluate_psl_diffs(arr, ax=None, plot=True):
    # Linear regression for the data, compare slope to y=x
    reg = LinearRegression().fit(arr[:, 0].reshape(-1, 1), arr[:, 1])
    y_pred_line = reg.predict(arr[:, 0].reshape(-1, 1))

    # Obtener la pendiente (slope) y la intersección (intercept)
    slope = reg.coef_[0]
    intercept = reg.intercept_

    # Mean squared error
    mse = mean_squared_error(arr[:, 0], arr[:, 1])

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        ax.plot(
            arr[:, 0],
            y_pred_line,
            color="red",
            label=f"Línea de regresión \n (y={slope:.2f}x + {intercept:.2f}) \n MSE: {mse:.2f}",
        )

        ax.scatter(arr[:, 0], arr[:, 1])

        # Plot y=x for the range of the data
        ax.plot(
            [arr[:, 0].min(), arr[:, 0].max()],
            [arr[:, 0].min(), arr[:, 0].max()],
            "k--",
            lw=2,
            label="x=y",
        )

        ax.set_xlabel("PSL real")
        ax.set_ylabel("PSL con Q estimado")

        ax.set_title(
            f"Comparación de PSL real y PSL con q(P1, P2) estimado - PSLs: {len(arr)} - Matches: {len(arr) / 2}"
        )

        # ax.legend()

    return slope, intercept, mse


# ## Models


def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    R_storage,
    sample_pairs,
    matches,
    player_features_df,
    ax=None,
):
    model.fit(np.array(X_train.values), np.array(y_train.values))

    # Predecir las probabilidades en los datos de prueba
    y_pred = model.predict(X_test)

    sc = model.score(X_test, y_test)
    q_mse = mean_squared_error(y_test, y_pred)

    display(Markdown(f"Score: {sc} - MSE: {q_mse}"))

    return model, y_pred, sc, q_mse


# Train the model with only passes of the player 1

test_matches = epl[epl["date"] >= date_2_3]
# Precompute and store features
player_features_df = precompute_and_store_features_optimized(test_matches)
# Save to disk
player_features_df.to_parquet("player_features_optimized.parquet")


player_features_df = load_player_features_dataframe("player_features_optimized.parquet")

player_features_df = precompute_and_store_features_optimized(
    epl[epl["date"] >= date_2_3]
)

sub_model_cols = ["pases/90_1", "pases/90_2"]
X_train_passes_1 = X_train[sub_model_cols]
X_train_passes_1["pases/90_2"] = 0
linreg_passes_1 = LinearRegression()


def model_info(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    R_storage,
    sample_pairs,
    matches,
    player_features_df,
):
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))

    model, y_pred, sc, q_mse = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        R_storage,
        sample_pairs,
        matches,
        player_features_df,
    )

    psl_slope, psl_intercept, psl_mse = evaluate_psl_diffs(
        np.array(
            get_psl_diffs(
                R_storage,
                model,
                sample_pairs,
                matches,
                player_features_df,
                list(set([col[:-2] for col in X_train.columns])),
            )
        ),
        ax=axs[1],
    )

    axs[1].legend()

    axs[0].scatter(y_test, y_pred)

    # Regresión lineal para los datos, comparar la pendiente con y=x
    reg = LinearRegression().fit(y_test.values.reshape(-1, 1), y_pred)
    y_pred_line = reg.predict(y_test.values.reshape(-1, 1))

    # Obtener la pendiente (slope) y la intersección (intercept)
    q_slope = reg.coef_[0]
    q_intercept = reg.intercept_

    axs[0].plot(
        y_test,
        y_pred_line,
        color="red",
        label=f"Línea de regresión \n (y={q_slope:.2f}x + {q_intercept:.2f})",
    )

    # Plot y=x for the range of the data
    axs[0].plot([y_test.min(), y_test.max()], [y_pred.min(), y_pred.max()], "k--")

    axs[0].set_xlabel("q real")
    axs[0].set_ylabel("q estimada")
    axs[0].set_title("Comparación de q estimada vs q real")

    axs[0].legend()

    if hasattr(model, "coef_"):
        _ = (
            pd.DataFrame(model.coef_, index=X_train.columns, columns=["importance"])
            .sort_values(by="importance")
            .plot(kind="barh", ax=axs[2])
        )
    # If model is xgb
    if isinstance(model, xgb.XGBRegressor):
        pd.DataFrame(
            model.feature_importances_, index=X_train.columns, columns=["importance"]
        ).sort_values(by="importance").plot(kind="barh", ax=axs[2])

    axs[2].set_title("Importancia de las variables")

    return model, y_pred, sc, q_mse, psl_slope, psl_intercept, psl_mse


model_info(
    linreg,
    X_train,
    y_train,
    X_test,
    y_test,
    R_storage,
    sample_pairs,
    matches,
    player_features_df,
)

model_info(
    DumbModel(),
    X_train,
    y_train,
    X_test,
    y_test,
    R_storage,
    sample_pairs,
    matches,
    player_features_df,
)

model_info(
    linreg_passes_1,
    X_train_passes_1,
    y_train,
    X_test[X_train_passes_1.columns],
    y_test,
    R_storage,
    sample_pairs,
    matches,
    player_features_df,
)

model_info(
    xgb_model,
    X_train,
    y_train,
    X_test,
    y_test,
    R_storage,
    sample_pairs,
    matches,
    player_features_df,
)

no_passes_cols = X_train.columns.difference(["pases/90_1", "pases/90_2"])
model_info(
    xgb.XGBRegressor(),
    X_train[no_passes_cols],
    y_train,
    X_test[no_passes_cols],
    y_test,
    R_storage,
    sample_pairs,
    matches,
    player_features_df,
)

# Remove a whole team from train
# Manchester City
mancity_id = epl[epl["home_team_name"] == "Manchester City"]["home_team_id"].values[0]
mancity_id

mancity_matches_id = epl[
    (epl["home_team_id"] == mancity_id) | (epl["away_team_id"] == mancity_id)
]["match_id"]

X_train_no_mancity = train_left[
    ~train_left["partido_id"].isin(mancity_matches_id)
].drop(columns=["player_1", "player_2", "target", "date", "partido_id"])

y_train_no_mancity = train_left[~train_left["partido_id"].isin(mancity_matches_id)][
    "target"
]

X_test_only_mancity = test_right[
    test_right["partido_id"].isin(mancity_matches_id)
].drop(columns=["player_1", "player_2", "target", "date", "partido_id"])

y_test_only_mancity = test_right[test_right["partido_id"].isin(mancity_matches_id)][
    "target"
]

mancity_matches = matches[matches["match_id"].isin(mancity_matches_id)]

# sub_R_storage = R_storage[mancity_matches.index]

_ = model_info(
    xgb.XGBRegressor(),
    X_train_no_mancity,
    y_train_no_mancity,
    X_test_only_mancity,
    y_test_only_mancity,
    R_storage,
    sample_pairs,
    mancity_matches,
    player_features_df,
)

# Remove half of the teams from train

team_ids = epl["home_team_id"].unique()

np.random.seed(42)
np.random.shuffle(team_ids)

half_teams = team_ids[: int(len(team_ids) / 2)]
half_teams_matches = matches[
    matches["home_team_id"].isin(half_teams) | matches["away_team_id"].isin(half_teams)
]

half_teams_matches_id = half_teams_matches["match_id"]

X_train_no_half_teams = train_left[
    ~train_left["partido_id"].isin(half_teams_matches_id)
].drop(columns=["player_1", "player_2", "target", "date", "partido_id"])

y_train_no_half_teams = train_left[
    ~train_left["partido_id"].isin(half_teams_matches_id)
]["target"]

X_test_only_half_teams = test_right[
    test_right["partido_id"].isin(half_teams_matches_id)
].drop(columns=["player_1", "player_2", "target", "date", "partido_id"])

y_test_only_half_teams = test_right[
    test_right["partido_id"].isin(half_teams_matches_id)
]["target"]

model_info(
    xgb.XGBRegressor(),
    X_train_no_half_teams,
    y_train_no_half_teams,
    X_test_only_half_teams,
    y_test_only_half_teams,
    R_storage,
    sample_pairs,
    half_teams_matches,
    player_features_df,
)


# Gradually destroy the R matrix, compute Q and PSL, and compare with the real PSL, MSE


def destroy_R(R, amount):
    # Destroy the R matrix by setting the amount of values to 0
    # Destroy the values in the diagonal
    # Return the destroyed R matrix

    R = R.copy()

    R = (np.random.rand(*R.shape) * amount) + (R * (1 - amount))

    return R


mse_over_amms = []

for amm in np.linspace(0, 1, 10):
    psl_diffs = []

    for i in range(R_storage.shape[0]):
        for j in range(R_storage.shape[1]):
            for k in range(R_storage.shape[2]):
                R = R_storage[i, j, k].copy()[1:, 1:]
                real_psl = psl_estimator(build_Q(R))

                mod_psl = psl_estimator(build_Q(destroy_R(R, amm)))

                psl_diffs.append((real_psl, mod_psl))

    psl_diffs_arr = np.array(psl_diffs)

    slope, intercept, mse = evaluate_psl_diffs(psl_diffs_arr, plot=False)

    mse_over_amms.append((amm, mse))

mse_over_amms = np.array(mse_over_amms)
