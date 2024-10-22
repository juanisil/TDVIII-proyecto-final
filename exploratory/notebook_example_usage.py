# %%

# %%
import sys

sys.path.append("../")

# %%
from src.bayesian_PSL import PlayerComparison, RandomVariablePSL, TeamPSL, EPL_Data

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
EPL_Full_Data = EPL_Data(
    "../SampleData/epl.xlsx", "../SampleData/players.json", "R_storage.npy"
)

# epl = EPL_Full_Data.get_epl()
# epl_player_data = EPL_Full_Data.get_epl_player_data()
# R_storage = EPL_Full_Data.get_r_storage()
# Q_storage = EPL_Full_Data.get_q_storage()
# partidos = EPL_Full_Data.get_partidos()
# tp_ds = EPL_Full_Data.get_transition_prob_dataset()
# player_ids = EPL_Full_Data.get_player_ids()
# player_kdes = EPL_Full_Data.get_player_kdes()

# %%

team_psl = TeamPSL(EPL_Full_Data)
team_psl.set_team("Manchester City")

player_comparison = PlayerComparison(team_psl)

sorted_rankings = player_comparison.rank_players(
    "Sergio Agüero",
    [
        "Olivier Giroud",
        "Wayne Rooney",
        "Romelu Lukaku",
    ],
)

sorted_rankings

# %%
og_list = team_psl.calculate_top_11_players()
psl_aguero = team_psl.estimate_psl_distribution(1000, og_list)
psl_giroud = team_psl.estimate_psl_distribution(
    1000,
    team_psl.replace_player("Sergio Agüero", "Olivier Giroud"),
)

# %%
import matplotlib.pyplot as plt

# %%
fig, ax = plt.subplots(figsize=(20, 8))

RandomVariablePSL.plot_psl_distributions(
    psl_aguero, psl_giroud, ax=ax, names=["Sergio Agüero", "Olivier Giroud"], colors=["b", "r"]
)

ax.set_xlim(0, 0.35)

# %%
from src.Player2Vec import Player2Vec, EPL_Graph

# %%
EPL_Full_Data

# %%
epl_graph = EPL_Graph(EPL_Full_Data)
epl_graph.build_graph()

# %%
p2v_model = Player2Vec(epl_data=EPL_Full_Data)
p2v_model.train(epl_graph.graph)

# %%
similars = p2v_model.get_most_similar("Sergio Agüero", 5)
similars

# %%
player_comparison.rank_players(
    "Sergio Agüero",
    similars,
)
