""" Module to load and access EPL player data from a JSON file. """

import json


class EPLPlayerData:
    """ Class to load and access EPL player data from a JSON file. """

    def __init__(self, json_path):
        self.json_path = json_path
        self.data = self._load_data()

    def _load_data(self):
        with open(self.json_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def get_player_info(self, player_opta_id):
        """ Get player information by OPTA ID.

        Args:
            player_opta_id (int): Player OPTA ID.

        Returns:
            dict: Player information.
        """
        for page in self.data:
            for player in page["content"]:
                if player["altIds"]["opta"] == f"p{player_opta_id}":
                    return player
        return None

    def get_player_name(self, player_opta_id):
        """ Get player name by OPTA ID.

        Args:
            player_opta_id (int): Player OPTA ID.

        Returns:
            str: Player name.
        """
        player_info = self.get_player_info(player_opta_id)
        # player_info["name"] => {'display': 'Lukas Podolski', 'first': 'Lukas', 'last': 'Podolski'}
        return player_info["name"]["display"] if player_info else None
    
    def get_player_id_by_lastname(self, lastname):
        """ Get player OPTA ID by player lastname.

        Args:
            lastname (str): Player lastname.

        Returns:
            int: Player OPTA ID.
        """
        for page in self.data:
            for player in page["content"]:
                if player["name"]["last"].lower() == lastname.lower():
                    return int(player["altIds"]["opta"][1:])
        return None
