import os
import pandas as pd


def separate_ccode(country_sector):
    return country_sector.split('_', maxsplit=1)


directory = "Datasets\\OECD_2016-2020"

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    year = separate_ccode(filename)[0]

    df = pd.read_csv(f)
    df.set_index('V1', inplace=True)

    # the last 3 rows are irrelevant for us
    num_of_entries = len(df.index) - 3
    # only taking country_sector into account
    country_relations = df.iloc[:num_of_entries, :num_of_entries]

    data = []
    for row in country_relations.iterrows():
        r_country_code, r_sector = separate_ccode(row[0])
        for item in row[1].items():
            i_country_code, i_sector = separate_ccode(item[0])
            data.append([year, r_country_code, r_sector, i_country_code, i_sector, item[1]])
    final_df = pd.DataFrame(data, columns=['Year', 'Exporter_country', 'Exporter_sector', 'Importer_country', 'Importer_sector', 'Value (million USD)'])
    final_df.to_csv(f"./Datasets/OECD_Transformed/OECD_{year}.csv", index=False)

    print(f"written file OECD_{year}.csv")


