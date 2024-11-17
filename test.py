import plotly.express as px
import pandas as pd

# Sample dictionaries
dict1 = {'A': 10, 'B': 20, 'C': 30}
dict2 = {'A': 15, 'B': 25, 'C': 35}
dict3 = {'A': 20, 'B': 30, 'C': 40}

# Create a DataFrame from the dictionaries
data = pd.DataFrame([dict1, dict2, dict3])

# Create the heatmap using Plotly Express
fig = px.imshow(data,
                 labels=dict(x="Categories", y="Dictionaries", color="Value"),
                 x=["A", "B", "C"],
                 y=["Dict1", "Dict2", "Dict3"])

fig.show()