import plotly.express as px

df = px.data.gapminder().query("year == 2007")

fig = px.scatter_geo(df,
    locations="iso_alpha", color="continent",
    hover_name="country", size="gdpPercap",
    projection="natural earth", title="PKB na mieszkańca w 2007")
fig.show()


import plotly.express as px
import pandas as pd
import seaborn as sns

df = sns.load_dataset("iris")  # cechy + klasy
corr = df.drop("species", axis=1).corr()

fig = px.imshow(corr,
                text_auto=True,
                title="Macierz korelacji dla zbioru Iris")
fig.show()


import plotly.express as px
df = px.data.iris()

fig = px.parallel_coordinates(df,
    color="species_id",
    dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
    color_continuous_scale=px.colors.diverging.Tealrose,
    title="Wielowymiarowe porównanie cech (Iris)")
fig.show()


import plotly.express as px

df = px.data.gapminder()
df = df[df['country'].isin(["Poland", "Germany", "France", "Italy", "Spain"])]

fig = px.line(df, x="year", y="gdpPercap", color="country",
              animation_frame="year", title="PKB na mieszkańca w czasie (animacja)",
              range_y=[0, 60000])
fig.show()


import plotly.express as px

# Dane Gapminder
df = px.data.gapminder()

# Animowany wykres: PKB na mieszkańca vs długość życia
fig = px.scatter(
    df, x="gdpPercap", y="lifeExp",
    animation_frame="year", animation_group="country",
    size="pop", color="continent", hover_name="country",
    log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90],
    title="Rozwój krajów: PKB vs Długość życia (Gapminder)"
)

fig.update_layout(transition={'duration': 500})
fig.show()

