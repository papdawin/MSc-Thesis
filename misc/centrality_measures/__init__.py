from .multilayer import *
from .singelayer import *
from .. import load_transformed_data_to_dataframe


sector_map = load_csv("./Datasets/sector_LUT.csv", "\ufeffID", "Code")
country_map = load_csv("./Datasets/country_LUT.csv", "\ufeffID", "Code")


def introductory_graph():
    df = load_transformed_data_to_dataframe("./Datasets/OECD_Transformed/OECD_1995.csv")
    visualize_multilayer_graph(df, sector_map, country_map)



def first_graph():
    pass


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


def degree_centrality():
    first_graph()
    second_graph()


