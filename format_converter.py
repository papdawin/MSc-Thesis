import os
import pandas as pd


def separate_ccode(country_sector):
    return country_sector.split('_', maxsplit=1)


directory = "Datasets\\OECD_2016-2020"
country_LUT = pd.read_csv("Datasets/country_LUT.csv", sep=';')
country_LUT = country_LUT[['ID', 'Code']]
country_LUT = country_LUT.set_index('Code')['ID'].to_dict()
sector_LUT = pd.read_csv("Datasets/sector_LUT.csv", sep=';')
sector_LUT = sector_LUT[['ID', 'Code']]
sector_LUT = sector_LUT.set_index('Code')['ID'].to_dict()


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
        r_country_code = country_LUT.get(r_country_code)
        r_sector = sector_LUT.get(r_sector)
        for item in row[1].items():
            i_country_code, i_sector = separate_ccode(item[0])
            i_country_code = country_LUT.get(i_country_code)
            i_sector = sector_LUT.get(i_sector)
            data.append([year, r_country_code, r_sector, i_country_code, i_sector, item[1]])
    transformed = pd.DataFrame(data, columns=['Year', 'Exporter_country', 'Exporter_sector', 'Importer_country', 'Importer_sector', 'Value (million USD)'])

    transformed.to_csv(f"./Datasets/OECD_Transformed/OECD_{year}.csv", index=False)

    print(f"written file OECD_{year}.csv")



