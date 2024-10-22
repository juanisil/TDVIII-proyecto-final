import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
from src.Player2Vec import Player2Vec
from src.bayesian_PSL import PlayerComparison, TeamPSL, EPL_Data, RandomVariablePSL

import warnings
warnings.filterwarnings("ignore")


EPL_DATA_PATH = "./SampleData/epl.xlsx"
PLAYERS_JSON_PATH = "./SampleData/players.json"
R_STORAGE_PATH = "./exploratory/R_storage.npy"
P2V_MODEL_PATH = "./exploratory/EPL_Graph_model_3.model"


# Configuración de la página
st.set_page_config(page_title="EPL Player Analysis", layout="wide")


# Cargar datos
@st.cache_data
def load_data():
    return EPL_Data(
        EPL_DATA_PATH,
        PLAYERS_JSON_PATH,
        R_STORAGE_PATH,
    )


EPL_Full_Data = load_data()


# Cargar el modelo Player2Vec guardado
@st.cache_data
def load_player2vec_model():
    return Player2Vec(model_path=P2V_MODEL_PATH, epl_data=EPL_Full_Data)


p2v_model = load_player2vec_model()

# Crear instancias de TeamPSL y PlayerComparison
team_psl = TeamPSL(EPL_Full_Data)
player_comparison = PlayerComparison(team_psl)

# Título de la aplicación
st.title("EPL Player Analysis")

# Sección de selección de equipo
st.sidebar.header("Select Team")
team_name = st.sidebar.selectbox("Team", EPL_Full_Data.get_all_teams())
team_psl.set_team(team_name)

# Sección de ranking de jugadores
st.header(f"Player Ranking Based on PSL KDE for {team_name}")
base_player_name = st.selectbox(
    "Base Player",
    EPL_Full_Data.get_player_names_for_team(team_name),
    format_func=lambda x: f"{x} ({EPL_Full_Data.get_epl_player_data().get_player_position((x_id := EPL_Full_Data.get_epl_player_data().get_player_id_by_name(x)))})",
)
compare_player_names = st.multiselect(
    "Compare Players",
    filter(lambda x: x != base_player_name, EPL_Full_Data.get_all_player_names()),
    format_func=lambda x: f"{x} ({EPL_Full_Data.get_epl_player_data().get_player_position((x_id := EPL_Full_Data.get_epl_player_data().get_player_id_by_name(x)))} - {EPL_Full_Data.get_player_team(x_id)})",
)

if st.button("Rank Players"):
    with st.spinner("Ranking players..."):
        sorted_rankings = player_comparison.rank_players(base_player_name, compare_player_names)
    st.write(
        pd.DataFrame(
            sorted_rankings,
            columns=["Player", f"P(PSL_KDE > PSL_KDE_{base_player_name})"],
        )
    )

# Sección de visualización de distribuciones de PSL
st.header("PSL Distributions")
player_1 = st.selectbox("Player 1", EPL_Full_Data.get_player_names_for_team(team_name))
player_2 = st.selectbox(
    "Player 2", filter(lambda x: x != player_1, EPL_Full_Data.get_all_player_names())
)

if st.button("Show PSL Distributions"):
    og_list = team_psl.calculate_top_11_players()
    psl_player_1 = team_psl.estimate_psl_distribution(1000, og_list)
    psl_player_2 = team_psl.estimate_psl_distribution(
        1000,
        team_psl.replace_player(player_1, player_2),
    )

    fig, ax = plt.subplots(figsize=(20, 8))
    p_greater = RandomVariablePSL.plot_psl_distributions(
        psl_player_1, psl_player_2, ax=ax, names=[player_1, player_2], colors=["b", "r"]
    )
    
    st.write(f"P(PSL_KDE_{player_2} > PSL_KDE_{player_1}) ≈ {p_greater}")
    
    ax.set_xlim(0, 0.50)
    st.pyplot(fig)


# Sección de jugadores similares utilizando Player2Vec
st.header("Similar Players Using Player2Vec")
player_name = st.selectbox("Player", EPL_Full_Data.get_player_names_for_team(team_name))
topn = st.slider("Number of Similar Players", 1, 20, 5)

if st.button("Find Similar Players"):
    similars = p2v_model.get_similar_players(player_name, topn)
    st.write(pd.DataFrame(similars, columns=["Player Name", "Player ID", "Team", "Similarity", "Position"]))
