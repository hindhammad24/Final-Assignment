import streamlit as st
import pickle
import pandas as pd
import plotly.express as px


# load model
model = pickle.load(
    open("pokemon_price_model.pkl", "rb")
)


# load model features
features = pickle.load(
    open("pokemon_features.pkl", "rb")
)


# load dataset
data = pd.read_csv("pokemon_cards.csv")


# page setup
st.set_page_config(
    page_title="Poké Pulls TCG",
    layout="wide"
)


# pokemon colors
blue = "#3B4CCA"
yellow = "#FFDE00"
red = "#FF0000"
dark = "#0B132B"


# app styling

st.markdown(
f"""
<style>

.stApp {{
    background-color: {dark};
}}


h1 {{
    color: white !important;
    font-size: 42px;
}}


h2 {{
    color: {yellow} !important;
}}


h3 {{
    color: white !important;
}}


p {{
    color: white !important;
}}


div[data-testid="stMetric"] {{
    background-color: grey;
    padding: 20px;
    border-radius: 15px;
}}


div[data-testid="stMetricValue"] {{
    color: {blue} !important;
    font-size: 32px;
}}


div[data-testid="stMetricLabel"] {{
    color: #333333 !important;
}}


.stButton button {{
    background-color: {red};
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}}


.stButton button:hover {{
    background-color: {yellow};
    color: black;
}}


section[data-testid="stSidebar"] {{
    background-color: grey;
}}


section[data-testid="stSidebar"] * {{
    color: #222222 !important;
}}


[data-testid="stMarkdownContainer"] p {{
    color: white !important;
}}


</style>
""",
unsafe_allow_html=True
)



# sidebar title

st.sidebar.title("card explorer")


st.sidebar.write(
"""
Explore Pokémon cards and
estimate market value.
"""
)



# title

st.title("Poké Pulls TCG")


st.subheader(
"predicting pokemon card market value"
)


st.write(
"""
This interactive dashboard explores how collectible features
such as rarity, artist, type, HP, and release year influence
Pokémon card prices.
"""
)



# overview metrics

st.header("dataset overview")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "total cards",
        len(data)
    )


with col2:

    st.metric(
        "average price",
        f"${data['market_price_usd'].mean():.2f}"
    )


with col3:

    st.metric(
        "highest value",
        f"${data['market_price_usd'].max():.2f}"
    )



# create tabs

overview, explore, predict, methodology = st.tabs(
[
"overview",
"explore data",
"predict price",
"methodology"
]
)



# overview tab

with overview:

    st.header("project overview")


    st.write(
    """
    Pokémon card prices vary greatly depending on collectible
    characteristics. Some cards remain affordable while others
    become valuable collectibles.

    This application uses machine learning to estimate card value
    and explore which features influence market prices.
    """
    )


    st.header("main findings")


    st.write(
    """
    - rarity showed the strongest relationship with card value.

    - artist and set information helped explain differences
    in collectible demand.

    - gameplay features such as HP had less influence compared
    to collectible features.
    """
    )

    # explore data tab

with explore:
    st.header("explore card market patterns")

    st.markdown(
        """
        ### understanding pokemon card value
        
        The value of a Pokémon card is influenced by many factors.
        These visualizations explore how collectible features such as
        rarity and artist information relate to market prices.

        The goal is to identify patterns in the dataset and understand
        which characteristics are connected to higher-value cards.
        """
    )



    # price distribution chart

    st.subheader("card price distribution")


    fig = px.histogram(
        data,
        x="market_price_usd",
        nbins=40,
        title="distribution of pokemon card prices",
        color_discrete_sequence=[blue]
    )


    fig.update_layout(
        template="plotly_white",
        xaxis_title="market price (usd)",
        yaxis_title="number of cards"
    )


    st.plotly_chart(
        fig,
        width="stretch",
        key="price_distribution"
    )


    st.info(
    """
    ### price distribution insights

    Most Pokémon cards in the dataset have relatively low market values.
    However, a small number of rare collectible cards reach much higher prices.

    This creates a heavily skewed price distribution, meaning that predicting
    card value is challenging because a few expensive cards can greatly affect
    the overall market range.
    """
    )



    # rarity chart

    st.subheader("rarity and market value")

    st.markdown(
    """
    ### does rarity influence price?

    Rarity represents how difficult a card is to obtain and is one of the
    main collectible features considered by collectors.

    This comparison uses median price because extreme high-value cards can
    affect the average price significantly.
    """
    )
    rarity_price = (
        data.groupby("rarity")["market_price_usd"]
        .median()
        .reset_index()
        .sort_values(
            "market_price_usd",
            ascending=False
        )
    )


    rarity_colors = [
        blue,
        yellow,
        red,
        "#7B2CBF",
        "#00A896",
        "#F77F00",
        "#457B9D"
    ]


    fig2 = px.bar(
        rarity_price,
        x="rarity",
        y="market_price_usd",
        title="median price by rarity",
        color="rarity",
        color_discrete_sequence=rarity_colors
    )


    fig2.update_layout(
        template="plotly_white",
        xaxis_title="rarity",
        yaxis_title="median price (usd)",
        showlegend=False
    )


    st.plotly_chart(
        fig2,
        width="stretch",
        key="rarity_chart"
    )

    st.markdown(
        """
        ### rarity insights

        Higher rarity categories generally show higher median prices.

        This supports the machine learning findings that collectible features,
        especially rarity, are important predictors of Pokémon card market value.
        """
    )



    # artist chart
    st.subheader("artist influence on card value")

    st.markdown(
    """
    ### does the illustrator matter?

    Artwork plays an important role in Pokémon card collecting.
    Certain artists may be associated with highly desired cards because
    of popularity, artistic style, or the cards they have contributed to.

    The chart shows the artists connected to the highest average card values
    in this dataset.
    """
    )

    artist_price = (
        data.groupby("artist")["market_price_usd"]
        .mean()
        .reset_index()
        .sort_values(
            "market_price_usd",
            ascending=False
        )
        .head(10)
    )


    fig3 = px.bar(
        artist_price,
        x="artist",
        y="market_price_usd",
        title="average price by artist",
        color="market_price_usd",
        color_continuous_scale="Viridis"
    )


    fig3.update_layout(
        template="plotly_white",
        xaxis_title="artist",
        yaxis_title="average price (usd)"
    )


    st.plotly_chart(
        fig3,
        width="stretch",
        key="artist_chart"
    )

    st.markdown(
        """
        ### artist insights

        Some artists are associated with higher average card prices.
        However, this relationship should be interpreted carefully because
        artist value can overlap with other factors such as rarity, set release,
        and collector demand.

        The model considers artist information as one feature among many when
        estimating card value.
        """
    )



# sidebar exploration filters

st.sidebar.divider()


st.sidebar.subheader("dataset explorer")


selected_rarity = st.sidebar.selectbox(
    "select rarity",
    sorted(data["rarity"].dropna().unique())
)


selected_type = st.sidebar.selectbox(
    "select type",
    sorted(data["types"].dropna().unique())
)



filtered = data[
    (data["rarity"] == selected_rarity)
    &
    (data["types"] == selected_type)
]


st.sidebar.write(
f"""
matching cards:

{len(filtered)}
"""
)



# prediction tab

with predict:

    st.header("estimate card market value")


    st.write(
    """
    Select card characteristics to estimate the expected
    market value using the trained machine learning model.
    """
    )



    # prediction inputs

    hp = st.sidebar.slider(
        "card hp",
        0,
        400,
        100
    )


    release_year = st.sidebar.selectbox(
        "release year",
        sorted(data["release_year"].dropna().unique())
    )


    prediction_rarity = st.sidebar.selectbox(
        "prediction rarity",
        sorted(data["rarity"].dropna().unique())
    )


    prediction_type = st.sidebar.selectbox(
        "prediction type",
        sorted(data["types"].dropna().unique())
    )


    prediction_artist = st.sidebar.selectbox(
        "prediction artist",
        sorted(data["artist"].dropna().unique())
    )



    if st.button("estimate price"):


        # create empty input row

        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=features
        )



        # add numeric values

        input_data["hp"] = hp

        input_data["release_year"] = release_year



        # add category values

        if f"rarity_{prediction_rarity}" in input_data.columns:

            input_data[
                f"rarity_{prediction_rarity}"
            ] = 1



        if f"types_{prediction_type}" in input_data.columns:

            input_data[
                f"types_{prediction_type}"
            ] = 1



        if f"artist_{prediction_artist}" in input_data.columns:

            input_data[
                f"artist_{prediction_artist}"
            ] = 1



        # make prediction

        prediction = model.predict(
            input_data
        )


        st.success(
            f"estimated market value: ${prediction[0]:.2f} usd"
        )


        st.write(
        """
        The prediction is based on patterns learned from previous
        Pokémon cards. Collectible features such as rarity and
        artist information have a major influence on card value.
        """
        )




# methodology tab

with methodology:

    st.header("machine learning approach")


    st.write(
    """
    Multiple regression models were evaluated:

    - linear regression

    - random forest regression

    - gradient boosting regression


    Gradient Boosting was selected because it captured complex
    relationships between card features and market prices.
    """
    )


    st.header("references")


    st.write(
    """
    Streamlit Documentation:
    https://streamlit.io/


    Streamlit GitHub:
    https://github.com/streamlit/streamlit
    """
    )