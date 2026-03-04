"""
Plotly chart helpers for the Streamlit app — Match Outcome Predictor.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

LABEL_ORDER = ["Away Win", "Draw", "Home Win"]
LABEL_COLORS = {
    "Home Win": "#4caf50",
    "Draw": "#ff9800",
    "Away Win": "#f44336",
}


def probability_bar(probabilities: dict, predicted_label: str) -> go.Figure:
    """Horizontal bar chart of match outcome probabilities."""
    labels = LABEL_ORDER
    values = [probabilities.get(l, 0) for l in labels]
    colors = [LABEL_COLORS.get(l, "#ccc") for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Predicted Match Outcome Probabilities",
        xaxis=dict(range=[0, 1.15], tickformat=".0%", title="Probability"),
        yaxis=dict(title=""),
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    # Highlight predicted label
    fig.add_vline(x=probabilities.get(predicted_label, 0), line_dash="dot", line_color="red")
    return fig


def sentiment_comparison(home_score: float, away_score: float, home_name: str, away_name: str) -> go.Figure:
    """Side-by-side bar chart comparing home and away media sentiment."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[home_name], y=[home_score],
        name=home_name, marker_color="#4caf50",
        text=[f"{home_score:.3f}"], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=[away_name], y=[away_score],
        name=away_name, marker_color="#f44336",
        text=[f"{away_score:.3f}"], textposition="outside",
    ))
    fig.update_layout(
        title="Pre-Match Media Sentiment",
        yaxis=dict(range=[-1.2, 1.2], title="Sentiment Score"),
        height=300,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    return fig


def team_comparison_chart(match_row: pd.Series, home_name: str, away_name: str) -> go.Figure:
    """Grouped bar chart comparing key team stats for home vs away."""
    metrics = {
        "Goals/Game": ("goals_per_game_home", "goals_per_game_away"),
        "Form": ("form_home", "form_away"),
        "League Position": ("league_position_home", "league_position_away"),
    }

    categories = []
    home_vals = []
    away_vals = []

    for label, (h_col, a_col) in metrics.items():
        if h_col in match_row.index and a_col in match_row.index:
            categories.append(label)
            home_vals.append(float(match_row.get(h_col, 0)))
            away_vals.append(float(match_row.get(a_col, 0)))

    if not categories:
        # Fallback to difference-based features
        diff_feats = ["elo_difference", "form_difference", "strength_ratio", "goal_difference_delta"]
        available = [f for f in diff_feats if f in match_row.index]
        return feature_importance_bar(
            {f: float(match_row.get(f, 0)) for f in available},
            top_n=len(available),
        )

    fig = go.Figure()
    fig.add_trace(go.Bar(name=home_name, x=categories, y=home_vals, marker_color="#4caf50"))
    fig.add_trace(go.Bar(name=away_name, x=categories, y=away_vals, marker_color="#f44336"))
    fig.update_layout(
        barmode="group",
        title="Team Comparison",
        yaxis_title="Value",
        height=350,
        margin=dict(l=10, r=10, t=60, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def feature_importance_bar(importances: dict, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart for feature importances."""
    sorted_items = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    features, scores = zip(*sorted_items) if sorted_items else ([], [])

    fig = go.Figure(go.Bar(
        x=list(scores), y=list(features), orientation="h",
        marker_color="#5c6bc0",
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance",
        yaxis=dict(autorange="reversed"),
        height=max(300, top_n * 22),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def ablation_comparison(results: list[dict]) -> go.Figure:
    """Grouped bar chart comparing model performance with/without NLP."""
    df = pd.DataFrame(results)
    fig = px.bar(
        df, x="model", y="f1_macro", color="suffix", barmode="group",
        color_discrete_map={"no_nlp": "#ef9a9a", "with_nlp": "#a5d6a7"},
        labels={"f1_macro": "F1-Macro", "model": "Model", "suffix": "Feature Set"},
        title="Ablation Study: F1-Macro With vs. Without NLP Features",
        text_auto=".3f",
    )
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=60, b=10))
    return fig
