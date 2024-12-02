import numpy as np
from flask import Flask
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from sympy import sympify, lambdify

# Flask App Setup
server = Flask(__name__)

# Dash App Setup
app = dash.Dash(__name__, server=server, url_base_pathname='/bisection/')


# Function to perform the Bisection Method
def bisection_method(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) >= 0:
        return None, None, "Function must have opposite signs at the endpoints a and b."

    iterations = []
    for i in range(max_iter):
        c = (a + b) / 2
        f_c = func(c)
        iterations.append({"Iteration": i + 1, "a": a, "b": b, "c": c, "f(c)": f_c})

        if abs(f_c) < tol or (b - a) / 2 < tol:
            return c, iterations, None  # Root found
        if f_c * func(a) < 0:
            b = c
        else:
            a = c

    return None, iterations, "Maximum iterations reached without convergence."


# Dash Layout
app.layout = html.Div(
    style={"fontFamily": "Arial", "margin": "30px", "backgroundColor": "#1e272e", "color": "#f5f6fa", "padding": "20px",
           "borderRadius": "15px"},
    children=[
        html.H1(
            "Precision Root Finder: Bisection Method Solver and Visualizer",
            style={
                "textAlign": "center",
                "color": "#f5f6fa",
                "marginBottom": "20px",
                "textShadow": "2px 2px 4px #000"
            }
        ),

        html.Div(
            style={
                "width": "60%", "margin": "0 auto", "padding": "20px",
                "borderRadius": "15px",
                "backgroundColor": "#2f3640",
                "boxShadow": "0px 6px 20px rgba(0, 0, 0, 0.4)"
            },
            children=[
                html.Label("Equation (in terms of x):", style={"fontWeight": "bold", "color": "#00a8ff"}),
                dcc.Input(
                    id="equation",
                    type="text",
                    placeholder="e.g., x**3 - 4*x + 1",
                    style={
                        "width": "100%",
                        "marginBottom": "15px",
                        "padding": "10px",
                        "borderRadius": "10px",
                        "border": "none",
                        "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.2)"
                    }
                ),

                html.Label("Lower Bound (a):", style={"fontWeight": "bold", "color": "#00a8ff"}),
                dcc.Input(
                    id="lower_bound",
                    type="number",
                    placeholder="e.g., 0",
                    style={
                        "width": "100%",
                        "marginBottom": "15px",
                        "padding": "10px",
                        "borderRadius": "10px",
                        "border": "none",
                        "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.2)"
                    }
                ),

                html.Label("Upper Bound (b):", style={"fontWeight": "bold", "color": "#00a8ff"}),
                dcc.Input(
                    id="upper_bound",
                    type="number",
                    placeholder="e.g., 2",
                    style={
                        "width": "100%",
                        "marginBottom": "15px",
                        "padding": "10px",
                        "borderRadius": "10px",
                        "border": "none",
                        "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.2)"
                    }
                ),

                html.Label("Tolerance:", style={"fontWeight": "bold", "color": "#00a8ff"}),
                dcc.Input(
                    id="tolerance",
                    type="number",
                    value=1e-6,
                    style={
                        "width": "100%",
                        "marginBottom": "15px",
                        "padding": "10px",
                        "borderRadius": "10px",
                        "border": "none",
                        "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.2)"
                    }
                ),

                html.Label("Maximum Iterations:", style={"fontWeight": "bold", "color": "#00a8ff"}),
                dcc.Input(
                    id="max_iter",
                    type="number",
                    value=100,
                    style={
                        "width": "100%",
                        "marginBottom": "20px",
                        "padding": "10px",
                        "borderRadius": "10px",
                        "border": "none",
                        "boxShadow": "0px 4px 8px rgba(0, 0, 0, 0.2)"
                    }
                ),

                html.Button(
                    "Solve",
                    id="solve_button",
                    n_clicks=0,
                    style={
                        "width": "100%",
                        "padding": "15px",
                        "borderRadius": "10px",
                        "backgroundColor": "#4cd137",
                        "color": "white",
                        "border": "none",
                        "fontWeight": "bold",
                        "boxShadow": "0px 6px 12px rgba(0, 0, 0, 0.3)",
                        "cursor": "pointer"
                    }
                ),
            ]
        ),

        html.Div(id="output_message",
                 style={"marginTop": "30px", "textAlign": "center", "fontSize": "20px", "color": "#00a8ff"}),

        html.Div(
            style={"display": "flex", "marginTop": "30px"},
            children=[
                dcc.Graph(id="bisection_graph",
                          style={"width": "60%", "marginRight": "20px", "boxShadow": "0px 6px 12px rgba(0, 0, 0, 0.3)",
                                 "borderRadius": "10px"}),
                dash_table.DataTable(
                    id="iteration_table",
                    columns=[
                        {"name": "Iteration", "id": "Iteration"},
                        {"name": "a", "id": "a"},
                        {"name": "b", "id": "b"},
                        {"name": "c (Midpoint)", "id": "c"},
                        {"name": "f(c)", "id": "f(c)"},
                    ],
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "center",
                        "padding": "10px",
                        "fontSize": "14px",
                        "border": "1px solid #ddd",
                        "backgroundColor": "#2f3640",
                        "color": "#f5f6fa"
                    },
                    style_header={"backgroundColor": "#00a8ff", "color": "white", "fontWeight": "bold"},
                )
            ]
        ),
    ]
)


# The rest of the code for callbacks remains the same.


# Dash Callback
@app.callback(
    [Output("output_message", "children"),
     Output("bisection_graph", "figure"),
     Output("iteration_table", "data")],
    [Input("solve_button", "n_clicks")],
    [Input("equation", "value"),
     Input("lower_bound", "value"),
     Input("upper_bound", "value"),
     Input("tolerance", "value"),
     Input("max_iter", "value")]
)
def update_output(n_clicks, equation, lower_bound, upper_bound, tolerance, max_iter):
    if not (equation and lower_bound and upper_bound):
        return "Please provide all inputs.", go.Figure(), []

    try:
        # Convert equation to a callable function
        x = sympify("x")
        func = lambdify(x, sympify(equation))

        # Apply the Bisection Method
        root, iterations, error = bisection_method(func, float(lower_bound), float(upper_bound), float(tolerance),
                                                   int(max_iter))

        if error:
            return error, go.Figure(), []

        # Prepare data for the graph
        a_vals = [row["a"] for row in iterations]
        b_vals = [row["b"] for row in iterations]
        c_vals = [row["c"] for row in iterations]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=a_vals, y=[func(a) for a in a_vals], mode="markers+lines", name="Lower Bound (a)"))
        fig.add_trace(go.Scatter(x=b_vals, y=[func(b) for b in b_vals], mode="markers+lines", name="Upper Bound (b)"))
        fig.add_trace(go.Scatter(x=c_vals, y=[func(c) for c in c_vals], mode="markers", name="Midpoint (c)",
                                 marker=dict(size=10, color="red")))

        fig.update_layout(
            title="Bisection Method Iterations",
            xaxis_title="x",
            yaxis_title="f(x)",
            legend_title="Bounds",
            template="plotly_white"
        )

        # Add a marker for the root
        fig.add_trace(go.Scatter(
            x=[root], y=[func(root)], mode="markers+text", name="Root",
            text=[f"Root: {root:.4f}"], textposition="top center", marker=dict(color="green", size=12)
        ))

        return f"Root: {root:.6f} (found in {len(iterations)} iterations)", fig, iterations

    except Exception as e:
        return f"Error: {e}", go.Figure(), []


if __name__ == "__main__":
    app.run_server(debug=True)
