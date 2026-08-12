import streamlit as st
import pickle
import pandas as pd
import plotly.express as px


# load saved model

model = pickle.load(
    open("pokemon_price_model.pkl", "rb")
)


# load model feature names

features = pickle.load(
    open("pokemon_features.pkl", "rb")
)


# load dataset for charts

data = pd.read_csv("pokemon_cards.csv")


# page setup

st.set_page_config(
    page_title="Poké Pulls TCG",
    layout="centered"
)


# pokemon colors

blue = "#3B4CCA"
yellow = "#FFDE00"
red = "#FF0000"


# simple pokemon styling
# simple pokemon styling

st.markdown(
f"""
<style>

/* main title */
[data-testid="stHeading"] h1 {{
    color: {blue};
}}


/* section headers */
[data-testid="stHeading"] h2 {{
    color: {blue};
}}


/* buttons */
.stButton > button {{
    background-color: {red};
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}}


.stButton > button:hover {{
    background-color: {yellow};
    color: black;
}}


/* info boxes */
[data-testid="stAlert"] {{
    border-left: 5px solid {yellow};
}}


</style>
""",
unsafe_allow_html=True
)

# title

st.title("Poké Pulls TCG")

st.subheader("Predicting Pokémon card market value")


st.write(
"""
This project explores whether Pokémon card prices can be predicted
using card information such as rarity, artist, type, HP, and release year.
"""
)



# about project

st.header("About the project")


st.write(
"""
Pokémon cards can have very different market values.
Some cards are common and inexpensive, while others become valuable
collectibles.

This application uses machine learning to understand which features
influence card prices and estimate the expected value of a card.
"""
)



# key findings

st.header("Key findings")


st.write(
"""
Rarity was one of the strongest factors connected to card value.

Artist and set information also showed important relationships with price.

Gameplay features such as HP had a smaller effect compared to collectible
features.
"""
)



# model section

st.header("Prediction model")


st.write(
"""
Different regression models were tested, including Linear Regression,
Random Forest, and Gradient Boosting.

Gradient Boosting was selected as the final model because it captured
complex patterns between card features and market price.
"""
)



# explore data

st.header("Explore the data")


st.write(
"""
These charts show patterns in the dataset and help explain
what influences Pokémon card value.
"""
)



# price distribution chart

st.subheader("How are card prices distributed?")

fig = px.histogram(
    data,
    x="market_price_usd",
    nbins=40,
    title="Distribution of Pokémon card prices",
    color_discrete_sequence=[blue]
)


fig.update_layout(
    xaxis_title="market price (USD)",
    yaxis_title="number of cards"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


st.write(
"""
Key finding:

Most cards have lower prices, while a small number of expensive cards
create a large difference in market value.

This uneven price distribution makes predicting card value more challenging.
"""
)



# rarity chart

st.subheader("Does rarity affect card value?")


rarity_price = (
    data.groupby("rarity")["market_price_usd"]
    .median()
    .reset_index()
    .sort_values(
        "market_price_usd",
        ascending=False
    )
)

fig2 = px.bar(
    rarity_price,
    x="rarity",
    y="market_price_usd",
    title="Median market price by rarity",
    color_discrete_sequence=[yellow]
)


fig2.update_layout(
    xaxis_title="rarity",
    yaxis_title="median price (USD)"
)


st.plotly_chart(
    fig2,
    use_container_width=True
)


st.write(
"""
Key finding:

Higher rarity levels usually have higher market values.

This supports the model result that collectible features
are important predictors of price.
"""
)



# prediction section

st.header("Predict a card value")


hp = st.slider(
    "card hp",
    0,
    400,
    100
)


release_year = st.selectbox(
    "release year",
    [
        2019,
        2020,
        2021,
        2022,
        2023,
        2024
    ]
)



rarity = st.selectbox(
    "rarity",
    [
        "Common",
        "Uncommon",
        "Rare",
        "Rare Holo",
        "Illustration Rare",
        "Special Illustration Rare",
        "Ultra Rare"
    ]
)



card_type = st.selectbox(
    "type",
    [
        "Colorless",
        "Darkness",
        "Dragon",
        "Fighting",
        "Fire",
        "Grass",
        "Lightning",
        "Metal",
        "Psychic",
        "Water"
    ]
)



artist = st.selectbox(
    "artist",
    [
        "Atsuko Nishida",
        "Mitsuhiro Arita",
        "Shinji Kanda",
        "NC Empire",
        "The Pokémon Company Art Team"
    ]
)



if st.button("predict price"):


    # create empty row

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=features
    )


    # add number values

    input_data["hp"] = hp

    input_data["release_year"] = release_year



    # add selected categories

    if f"rarity_{rarity}" in input_data.columns:
        input_data[f"rarity_{rarity}"] = 1


    if f"types_{card_type}" in input_data.columns:
        input_data[f"types_{card_type}"] = 1


    if f"artist_{artist}" in input_data.columns:
        input_data[f"artist_{artist}"] = 1



    # predict

    prediction = model.predict(input_data)



    st.success(
        f"Estimated market value: ${prediction[0]:.2f} USD"
    )


    st.caption(
        "This estimate is based on patterns learned from previous Pokémon cards in the dataset."
    )


    st.write(
    """
    The prediction is mainly influenced by collectible features
    such as rarity, artist information, and set characteristics.
    """
    )



# references

st.header("References")


st.write(
"""
This application was created using Streamlit.

Streamlit Documentation:
https://streamlit.io/

Streamlit GitHub:
https://github.com/streamlit/streamlit
"""
)