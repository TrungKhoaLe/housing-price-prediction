from sqlalchemy import create_engine, text
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
from pathlib import Path
import joblib
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.tree import plot_tree, export_graphviz
import xml
import pydot
import base64


def connect_read_sql(query, engine):
    result = pd.read_sql(query, engine)
    return result


def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = 42

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

basedir = Path(__file__).resolve().parent
engine = create_engine("sqlite:///" + f"{basedir}" + "/data.sqlite",
                       echo=False)
feature_names = joblib.load(f"{basedir}" + "/feature_names.joblib")


app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Machine Learning App with Dash"
server = app.server


# Models
MODELS = ("model-linear", "model-decision-tree", "model-random-forest")

query_card = dbc.Card(
        [
            dbc.CardHeader("Native SQL Query"),
            dbc.CardBody(dcc.Markdown(id="sql-query"))
        ])

query_area = dbc.InputGroup(
            [
                dbc.InputGroupText("Native SQL Query"),
                dbc.Textarea(id="sql-query-area",
                             value="select * from new_info"),
            ])

submit_btn = dbc.Button("Submit",
                        id="submit-btn",
                        className="me-1",
                        color="success",
                        n_clicks=0)

sample_query = [
       dbc.Col(query_area),
       dbc.Col(submit_btn, width={"offset": 4})
]


output_card = dbc.Card(
        [
            dbc.CardHeader("Real time Prediction"),
            dbc.CardBody(html.H2(id="prediction-score", style={"text-align": "center"}))
        ])

"""
table_selection = dbc.InputGroup(
        [
            dbc.InputGroupText("Select Table"),
            dbc.Select(
                id="table-selection",
                options=[{"label": t, "value": t} for t in ["RENT",
                "MORTGAGE", "OWNER"]],
                value="RENT")
                                
        ])
"""

model_selection = dbc.InputGroup(
        [
            dbc.InputGroupText("Select Model"),
            dbc.Select(
                id="model-selection",
                options=[
                            {
                                "label": m.replace("-", " ").capitalize(),
                                "value": "assets/" + m + ".joblib"
                            }
                            for m in MODELS
                        ],
                value="assets/" + MODELS[1] + ".joblib"
                )
        ])

nav_buttons = dbc.ButtonGroup(
        [
            dbc.Button(
                "Prev. Sample",
                id="prev-sample",
                color="success",
                n_clicks=0,
                outline=True),
            dbc.Button(
                "Next Sample",
                id="next-sample",
                color="success",
                n_clicks=0)
        ],
        style={"width": "100%"})
"""
sample_controls = [
                    dbc.Col(table_selection), 
                    dbc.Col(nav_buttons)
                  ]
"""

controls = [dbc.Col(nav_buttons, width={"size": 3, "offset": 4}),dbc.Col(model_selection, md=5)]


app.layout = dbc.Container(
                children=[
                    dcc.Store(id="store"),
                    html.H1("PaveML"),
                    html.Hr(),
                    dbc.Row(controls, style={"padding": "20px 0px"}),
                    dbc.Row(
                        [
                            dbc.Col(
                                children=[
                                            dbc.Row([dbc.Col(sample_query), dbc.Col(output_card)]),
                                            dash_table.DataTable(
                                                id="table-sample",
                                                style_table={
                                                    "height": "500px",
                                                    "overflowY": "auto",
                                                    "padding": "20px",
                                                }),
                                         ],
                                md=7
                                ),
                            dbc.Col(
                                dcc.Graph(
                                    id="graph-tree",
                                    style={"height": "700px"}),
                                md=5)
                        ])
                ],
                fluid=True,
                style={"margin": "auto"}
             )


@app.callback(
    Output("store", "data"),
    Input("submit-btn", "n_clicks"),
    State("sql-query-area", "value")
)
def query_and_store(n_click, query):
    query_df = connect_read_sql(query, engine)
    return query_df.to_json()


@app.callback(
    Output("table-sample", "data"),
    Output("table-sample", "columns"),
    Output("prediction-score", "children"),
    Input("store", "data"),
    Input("prev-sample", "n_clicks"),
    Input("next-sample", "n_clicks"),
    Input("model-selection", "value"),
)
def generate_table(query_json, prev_clicks, next_clicks, model_path):
    query_df = pd.read_json(query_json)
    feature_df = query_df.drop("median_house_value", axis=1)

    # build the sample table
    i = max(0, next_clicks - prev_clicks)
    table_df = feature_df.loc[i:i].T.reset_index()
    table_df.columns = ["Feature", "Current Value"]

    # load model and make prediction
    model = joblib.load(model_path)
    model_input = feature_df.loc[i:i]
    pred = model.predict(model_input)[0]

    columns = [{"name": i, "id": i} for i in table_df.columns]

    return table_df.to_dict("records"), columns, f"Price = ${pred: .2f}"


if __name__ == "__main__":
    app.run_server(debug=True)
