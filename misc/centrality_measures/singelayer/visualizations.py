import pandas as pd
import networkx as nx
import plotly.express as px


def createdf(DC, year):
    labels = []
    parents = []
    values = []
    for el in DC:
        labels.append(el[4:])
        parents.append(el[:3])
        values.append(DC[el])

    df = pd.DataFrame({"parents": parents,
                       "labels": labels,
                       "dcvalues": values})
    df['year'] = year
    return df

def plot_by_continents(earlyDC, lateDC, firstyear, lastyear):
    fdf = createdf(earlyDC, firstyear)
    sdf = createdf(lateDC, lastyear)

    df = fdf._append(sdf, ignore_index=True)

    north_america = ["CAN", "CRI", "MEX", "USA", "ARG", "BRA", "CHL", "COL", "PER"]
    europe = [
        "AUT", "BEL", "BGR", "BLR", "CHE", "CYP", "CZE", "DEU", "DNK", "ESP", "EST",
        "FIN", "FRA", "GBR", "GRC", "HRV", "HUN", "IRL", "ISL", "ITA", "LTU", "LUX",
        "LVA", "MLT", "NLD", "NOR", "POL", "PRT", "ROU", "RUS", "SVK", "SVN", "SWE",
        "UKR"
    ]
    asia = [
        "BGD", "BRN", "CHN", "HKG", "IDN", "IND", "ISR", "JPN", "JOR", "KAZ", "KHM",
        "KOR", "LAO", "MMR", "MYS", "PAK", "PHL", "SAU", "SGP", "THA", "TUR", "TWN",
        "VNM"
    ]
    africa = ["CIV", "CMR", "EGY", "MAR", "NGA", "SEN", "TUN", "ZAF"]
    oceania = ["AUS", "NZL"]
    for continent in [north_america, europe, asia, africa, oceania]:
        df_cont = df[df['parents'].isin(continent)]
        fig = px.box(df_cont, x="parents", y="dcvalues", color="year", labels={"parents": "County", "dcvalues": "Degree centrality values"})
        fig.show()
