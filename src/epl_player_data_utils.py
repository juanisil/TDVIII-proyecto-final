""" Module to load and access EPL player data from a JSON file. """

import json


SCHEMA = """ [
  {
    "pageInfo": {
      "page": 5,
      "numPages": 97,
      "pageSize": 10,
      "numEntries": 965
    },
    "content": [
      {
        "playerId": 51322,
        "info": {
          "position": "D",
          "shirtNum": 17,
          "positionInfo": "Centre Central Defender"
        },
        "nationalTeam": {
          "isoCode": "GB-ENG",
          "country": "England",
          "demonym": "English"
        },
        "previousTeam": {
          "name": "Bristol City",
          "club": { "name": "Bristol City", "abbr": "BCI", "id": 132 },
          "teamType": "FIRST",
          "shortName": "Bristol City",
          "id": 132,
          "altIds": { "opta": "t113" }
        },
        "birth": {
          "date": { "millis": 672364800000, "label": "23 April 1991" },
          "country": {
            "isoCode": "GB-ENG",
            "country": "England",
            "demonym": "English"
          }
        },
        "age": "33 years 122 days",
        "name": {
          "display": "Nathan Baker",
          "first": "Nathan",
          "last": "Baker"
        },
        "id": 3561,
        "altIds": { "opta": "p52477" }
      }, """


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

    def get_player_position(self, player_opta_id):
        """ Get player position by OPTA ID.

        Args:
            player_opta_id (int): Player OPTA ID.

        Returns:
            str: Player position.
        """
        player_info = self.get_player_info(player_opta_id)
        return player_info["info"]["position"] if player_info else None

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

    def __len__(self):
        return sum(len(page["content"]) for page in self.data)

    def __repr__(self):
        return f"EPLPlayerData({self.json_path}): {len(self)} players"
