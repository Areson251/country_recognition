"""
---------------------------
Config & Country metadata
---------------------------
"""

# Сountry list with aliases
COUNTRY_LIST: dict[str, list[str]] = {
    "BEL": ["Belgium", "Belgique", "België", "Бельгия"],
    "BGR": ["Bulgaria", "България"],
    "BLR": ["Belarus", "Беларусь", "Белоруссия"],
    "CAN": ["Canada", "Kanada", "Канада"],
    "CHL": ["Chile", "Чили"],
    "DEU": ["Germany", "Deutschland", "Германия"],
    "DOM": ["Dominican Republic", "República Dominicana", "Доминиканская Республика"],
    "ESP": ["Spain", "España", "Испания"],
    "EST": ["Estonia", "Eesti", "Эстония"],
    "GBR": ["United Kingdom", "UK", "Great Britain", "Britain", "England", "Великобритания", "Англия"],
    "HUN": ["Hungary", "Magyarország", "Венгрия"],
    "IDN": ["Indonesia", "Indonesia", "Индонезия"],
    "IRL": ["Ireland", "Éire", "Ирландия"],
    "ITA": ["Italy", "Italia", "Италия"],
    "KAZ": ["Kazakhstan", "Қазақстан", "Казахстан"],
    "KGZ": ["Kyrgyzstan", "Кыргызстан", "Киргизия"],
    "MDA": ["Moldova", "Republica Moldova", "Молдова"],
    "MEX": ["Mexico", "México", "Мексика"],
    "NLD": ["Netherlands", "Nederland", "Голландия", "Нидерланды"],
    "POL": ["Poland", "Polska", "Польша"],
    "SVK": ["Slovakia", "Slovensko", "Словакия"],
    "SWE": ["Sweden", "Sverige", "Швеция"],
    "USA": ["United States", "USA", "US", "United States of America", "Америка", "США"],
    "UZB": ["Uzbekistan", "Oʻzbekiston", "Узбекистан"],
}

# MRZ issuing state (ISO 3166-1 alpha-3) map for the 24 countries
ALPHA3_MAP: dict[str, str] = {
    code: names[0] for code, names in COUNTRY_LIST.items()
}

COUNTRIES = list(COUNTRY_LIST.keys())
NUM_CLASSES = len(COUNTRIES)
NAME_TO_IDX = {n: i for i, n in enumerate(COUNTRIES)}
IDX_TO_NAME = {i: n for n, i in NAME_TO_IDX.items()}