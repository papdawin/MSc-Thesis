from .multilayer import visualize_multilayer_graph, get_network_country_yearly, get_bc_by_year, get_hc_by_year, \
    get_eb_by_year, get_density_by_year_by_country
from .singelayer import get_pr_by_year, get_density_by_year
from .. import load_transformed_data_to_dataframe, load_csv
from plotly.subplots import make_subplots

sector_map = load_csv("./Datasets/sector_LUT.csv", "\ufeffID", "Code")
country_map = load_csv("./Datasets/country_LUT.csv", "\ufeffID", "Code")


def introductory_graph():
    df = load_transformed_data_to_dataframe("./Datasets/OECD_Transformed/OECD_1995.csv")
    visualize_multilayer_graph(df, sector_map, country_map)


def first_graph():
    df = load_transformed_data_to_dataframe("./Datasets/OECD_Original/ICIO2023_1995.csv")
    df2 = load_transformed_data_to_dataframe("./Datasets/OECD_Original/ICIO2023_2020.csv")
    compare_degree_centralities(df, df2)


def second_graph():
    resdf = []
    for i in range(26):
        transformed_df = load_transformed_data_to_dataframe(f"./Datasets/OECD_Transformed/OECD_{1995 + i}.csv")
        year_val = get_network_country_yearly(transformed_df)
        resdf.append(year_val)
    data = pd.DataFrame(resdf)
    data = data.reindex(sorted(data.columns), axis=1)
    new_index = {i: str(i + 1995) for i in data.index}
    data = data.rename(index=new_index)
    data = data.rename(columns=country_map)
    fig = px.imshow(data, aspect="auto")
    fig.show()

import pandas as pd
import plotly.graph_objects as go


def third_graph():
    resdf = []
    for year in range(1995, 2021):
        bc = get_bc_by_year(year)
        resdf.append(bc)
    df = pd.DataFrame(resdf)
    df = df.rename(columns=country_map)

    # brics_countries = ["BRA", "RUS", "IND", "CHN", "SAU"]
    # df = df[brics_countries]
    # print(df)
    colonizers = ["ESP", "FRA", "GBR", "PRT", "NLD", "BEL", "DEU", "ITA", "USA", "JPN", "CHN"]
    non_colonizer_cols = [col for col in df.columns if col not in colonizers + ["WXD", "ROW"]]
    df['Average'] = df[non_colonizer_cols].mean(axis=1)
    df_n = df[colonizers]
    df_n['Average'] = df['Average']
    df_n.index += 1995

    fig = px.line(df_n, x=df_n.index, y=df_n.columns, title='Former colonizer countries',)
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Betweenness centrality value",
        legend_title="Country"
    )
    fig.show()


def fourth_graph():
    bc = get_hc_by_year(1995)
    country_bc = {country_map[key]: value for key, value in bc.items()}
    df_1995 = pd.DataFrame(list(country_bc.items()), columns=["country_code", "bc_value"])
    bc = get_hc_by_year(2020)
    country_bc = {country_map[key]: value for key, value in bc.items()}
    df_2020 = pd.DataFrame(list(country_bc.items()), columns=["country_code", "bc_value"])

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=['Harmonic centrality in 1995', 'Harmonic centrality in 2020'],
        specs=[[{'type': 'choropleth'}, {'type': 'choropleth'}]],
        horizontal_spacing=0.05
    )
    fig.add_trace(px.choropleth(df_1995, locations="country_code",
                        color="bc_value",
                        hover_name="country_code").data[0], row=1, col=1)
    fig.add_trace(px.choropleth(df_2020, locations="country_code",
                                color="bc_value",
                                hover_name="country_code").data[0], row=1, col=2)
    fig.show()


import plotly.express as px


def fifth_graph():
    resdf = []
    for year in range(1995, 2021):
        pr = get_pr_by_year(year)
        pr = pd.Series(pr)
        top_100 = pr.nlargest(50)
        resdf.append(top_100)
    df = pd.DataFrame(resdf)
    melted_df = pd.melt(df, var_name='variable', value_name='value')
    melted_df['country'] = melted_df['variable'].str[:3]
    melted_df['category'] = melted_df['variable'].str[4:]
    years = list(range(1995, 2021)) * (len(melted_df) // len(range(1995, 2021)) + 1)
    melted_df['year'] = years[:len(melted_df)]
    countries = melted_df['country'].unique()
    sectors = melted_df['category'].unique()
    fig = px.bar(melted_df, x="year", y="value", color="country", text="category")
    fig.update_layout(title_text="Top 50 PageRank Value Sectors")
    fig.show()


def sixth_graph():
    bc = get_eb_by_year(1995)
    print(bc)
    country_bc = {country_map[key]: value for key, value in bc.items()}
    df_1995 = pd.DataFrame(list(country_bc.items()), columns=["country_code", "bc_value"])
    print(df_1995)


def seventh_graph():
    resdf = []
    year_range = list(range(1995, 2021))
    for year in year_range:
        pr = get_density_by_year(year, filter_insignificant=False)
        resdf.append(pr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=year_range, y=resdf, mode='lines+markers', name='Network density'))
    fig.show()


def eighth_graph():
    resdf = []
    year_range = list(range(1995, 2021))
    country_list = [10, 13, 20, 26, 27, 32, 33, 37, 39, 42]
    for year in year_range:
        subres = []
        for country in country_list:
            density = get_density_by_year_by_country(year, country)
            subres.append(density)
        resdf.append(subres)
    df = pd.DataFrame(resdf)
    print(df)
    df_transposed = df.T
    print(df_transposed)
    fig = go.Figure()
    for idx, year_df in df_transposed.iterrows():
        fig.add_trace(go.Scatter(x=year_range, y=year_df, mode='lines+markers', name=country_map[country_list[idx]]))
    fig.show()


def ninth_graph():
    resdf = []
    year_range = list(range(1995, 2021))
    # sector_list = [9, 17, 18, 19, 20, 21, 23, 24, 25, 34]
    sector_list = list(range(1, 46))
    for year in year_range:
        subres = []
        for country in sector_list:
            density = get_density_by_year_by_country(year, country)
            subres.append(density)
        resdf.append(subres)
    df = pd.DataFrame(resdf)
    df_transposed = df.T
    fig = go.Figure()
    for idx, year_df in df_transposed.iterrows():
        fig.add_trace(go.Scatter(x=year_range, y=year_df, mode='lines+markers', name=sector_map[sector_list[idx]]))
    fig.show()


def tenth_graph():
    pass