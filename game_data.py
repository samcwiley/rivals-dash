# stages
class Stage:
    def __init__(self, name, width, side_blast, top_blast, bot_blast):
        self.name = name
        self.width = width
        self.side_blast = side_blast
        self.top_blast = top_blast
        self.bot_blast = bot_blast

    def __repr__(self):
        return (
            f"Stage(Name='{self.name}', Width={self.width}, "
            f"Side_Blast={self.side_blast}, Top_Blast={self.top_blast}, Bot_Blast={self.bot_blast})"
        )


stages = {
    "Aetherean Forest": Stage(
        name="Aetherean Forest",
        width=1280,
        side_blast=1380,
        top_blast=2180,
        bot_blast=1315,
    ),
    "Godai Delta": Stage(
        name="Godai Delta", width=1600, side_blast=1680, top_blast=2230, bot_blast=1280
    ),
    "Hodojo": Stage(
        name="Hodojo", width=1460, side_blast=1478, top_blast=2195, bot_blast=1330
    ),
    "Julesvale": Stage(
        name="Julesvale", width=1400, side_blast=1560, top_blast=2170, bot_blast=1360
    ),
    "Merchant Port": Stage(
        name="Merchant Port",
        width=1630,
        side_blast=1540,
        top_blast=2140,
        bot_blast=1305,
    ),
    "Air Armada": Stage(
        name="Air Armada", width=1900, side_blast=1630, top_blast=2080, bot_blast=1290
    ),
    "Fire Capital": Stage(
        name="Fire Capital", width=2020, side_blast=1814, top_blast=2254, bot_blast=1470
    ),
    "Hyperborean Harbor": Stage(
        name="Hyperborean Harbor",
        width=1560,
        side_blast=1840,
        top_blast=1940,
        bot_blast=1390,
    ),
    "Rock Wall": Stage(
        name="Rock Wall", width=1860, side_blast=1320, top_blast=2130, bot_blast=1420
    ),
    "Tempest Peak": Stage(
        name="Tempest Peak", width=1250, side_blast=1645, top_blast=2260, bot_blast=1250
    ),
}
starter_stages = [
    "Aetherean Forest",
    "Godai Delta",
    "Hodojo",
    "Julesvale",
    "Merchant Port",
]
counter_stages = [
    "Air Armada",
    "Fire Capital",
    "Hyperborean Harbor",
    "Rock Wall",
    "Tempest Peak",
]
all_stages = starter_stages + counter_stages
characters = [
    "Clairen",
    "Fleet",
    "Forsburn",
    "Kragg",
    "Loxodont",
    "Maypul",
    "Orcane",
    "Ranno",
    "Wrastor",
    "Zetterburn",
]
character_icons = {
    "Clairen": "character_icons/clairen.png",
    "Fleet": "character_icons/fleet.png",
    "Forsburn": "character_icons/forsburn.png",
    "Kragg": "character_icons/kragg.png",
    "Loxodont": "character_icons/loxodont.png",
    "Maypul": "character_icons/maypul.png",
    "Orcane": "character_icons/orcane.png",
    "Ranno": "character_icons/ranno.png",
    "Wrastor": "character_icons/wrastor.png",
    "Zetterburn": "character_icons/zetterburn.png",
    "Multiple": "character_icons/rivals.png",
}
