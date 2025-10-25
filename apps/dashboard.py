"""Streamlit dashboard for NFL Data Bowl 2025 player tracking visualization."""

import streamlit as st
import polars as pl
import plotly.graph_objects as go
from pathlib import Path
import random

# Constants
DATA_DIR = Path("data/train")
FIELD_LENGTH = 120
FIELD_WIDTH = 53.3
TEAM_COLORS = {"Offense": "#0066CC", "Defense": "#CC0000"}


@st.cache_data
def load_data(week_num):
    """Load input and output data for a given week."""
    input_file = DATA_DIR / f"input_2023_w{week_num:02d}.csv"
    output_file = DATA_DIR / f"output_2023_w{week_num:02d}.csv"

    input_df = pl.read_csv(input_file)
    output_df = pl.read_csv(output_file)

    return input_df, output_df


def get_unique_games(df):
    """Extract unique game_ids from dataframe."""
    return sorted(df["game_id"].unique().to_list())


def get_plays_for_game(df, game_id):
    """Get all play_ids for a given game."""
    return sorted(df.filter(pl.col("game_id") == game_id)["play_id"].unique().to_list())


def load_predictions(pred_file):
    """Load prediction file if provided."""
    if pred_file is not None:
        return pl.read_csv(pred_file)
    return None


def create_field_plot():
    """Create base football field plot."""
    fig = go.Figure()

    # Field boundaries - clean white with subtle green tint
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=FIELD_LENGTH,
        y1=FIELD_WIDTH,
        line=dict(color="darkgreen", width=2),
        fillcolor="rgba(245, 255, 250, 0.5)",  # Very light mint
    )

    # Yard lines
    for x in range(10, FIELD_LENGTH, 10):
        fig.add_shape(
            type="line",
            x0=x,
            y0=0,
            x1=x,
            y1=FIELD_WIDTH,
            line=dict(color="rgba(100, 149, 237, 0.3)", width=1, dash="dot"),
        )

    fig.update_layout(
        width=1200,
        height=400,
        xaxis=dict(range=[-5, FIELD_LENGTH + 5], showgrid=False, zeroline=False),
        yaxis=dict(range=[-5, FIELD_WIDTH + 5], showgrid=False, zeroline=False),
        plot_bgcolor="white",
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def add_traces_to_plot(fig, input_df, output_df, pred_df=None):
    """Add player movement traces to the field plot."""

    # Group by player with player_to_predict flag
    players = input_df.select(
        ["nfl_id", "player_name", "player_side", "player_to_predict", "player_role"]
    ).unique()

    for row in players.iter_rows(named=True):
        nfl_id = row["nfl_id"]
        name = row["player_name"]
        side = row["player_side"]
        to_predict = row["player_to_predict"]
        player_role = row["player_role"]
        color = TEAM_COLORS[side]

        # Pre-throw trajectory (input) - lighter and thinner
        player_input = input_df.filter(pl.col("nfl_id") == nfl_id).sort("frame_id")
        x_input = player_input["x"].to_list()
        y_input = player_input["y"].to_list()

        # Post-throw trajectory (output) - bolder and thicker
        player_output = output_df.filter(pl.col("nfl_id") == nfl_id).sort("frame_id")
        x_output = player_output["x"].to_list()
        y_output = player_output["y"].to_list()

        # Adjust sizes for players to predict - keep markers small
        marker_size_pre = 3
        marker_size_post = 5
        line_width_pre = 1.5
        line_width_post = 3.5 if to_predict else 2

        # Add pre-throw trace (lighter, thinner, dashed)
        if x_input:
            fig.add_trace(
                go.Scatter(
                    x=x_input,
                    y=y_input,
                    mode="lines+markers",
                    name=f"{name} (PRE-THROW)",
                    line=dict(color=color, width=line_width_pre, dash="dash"),
                    marker=dict(
                        size=marker_size_pre,
                        color=color,
                        opacity=0.6,
                        line=dict(width=0),
                    ),
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate=f"<b>{name}</b> [PRE-THROW]<br>{side} - {player_input['player_position'][0]}<br>Frame: %{{pointNumber}}<br>x=%{{x:.2f}}, y=%{{y:.2f}}<extra></extra>",
                )
            )

            # Add orientation vector at end of pre-throw
            last_frame = player_input.filter(
                pl.col("frame_id") == player_input["frame_id"].max()
            )
            if len(last_frame) > 0:
                x_end = last_frame["x"][0]
                y_end = last_frame["y"][0]
                movement_dir = last_frame["dir"][0]  # direction of motion in degrees

                # Convert orientation to radians and calculate arrow endpoint
                # Direction: 0° = North (+y), 90° = East (+x), clockwise
                import math

                dir_rad = math.radians(movement_dir)
                arrow_length = 2.5  # yards
                x_arrow = x_end + arrow_length * math.sin(dir_rad)
                y_arrow = y_end + arrow_length * math.cos(dir_rad)

                # Use line shape for cleaner arrow
                fig.add_shape(
                    type="line",
                    x0=x_end,
                    y0=y_end,
                    x1=x_arrow,
                    y1=y_arrow,
                    line=dict(color=color, width=2),
                    opacity=0.7,
                )

                # Add small arrowhead as a triangle marker
                fig.add_trace(
                    go.Scatter(
                        x=[x_arrow],
                        y=[y_arrow],
                        mode="markers",
                        marker=dict(
                            size=6,
                            color=color,
                            symbol="triangle-up",
                            angle=movement_dir,
                            line=dict(width=0),
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                        opacity=0.7,
                    )
                )

                # Mark the passer (ball thrower) with special marker
                if player_role == "Passer":
                    fig.add_trace(
                        go.Scatter(
                            x=[x_end],
                            y=[y_end],
                            mode="markers+text",
                            name="QB/Passer",
                            marker=dict(
                                size=15,
                                color=color,
                                symbol="square",
                                line=dict(color="black", width=2),
                            ),
                            text=["QB"],
                            textposition="top center",
                            textfont=dict(size=10, color="black", family="Arial Black"),
                            showlegend=False,
                            hovertemplate=f"<b>{name} (PASSER)</b><br>Ball release position<br>x={x_end:.2f}, y={y_end:.2f}<extra></extra>",
                        )
                    )

        # Add post-throw trace (bolder, thicker, solid)
        if x_output:
            legend_suffix = " ⭐ TO PREDICT" if to_predict else ""
            player_role = (
                player_input["player_role"][0] if len(player_input) > 0 else ""
            )
            fig.add_trace(
                go.Scatter(
                    x=x_output,
                    y=y_output,
                    mode="lines+markers",
                    name=f"{name} ({side}){legend_suffix}",
                    line=dict(
                        color=color,
                        width=line_width_post,
                        dash="solid" if not to_predict else "dashdot",
                    ),
                    marker=dict(
                        size=marker_size_post,
                        color=color,
                        line=dict(width=1 if to_predict else 0, color="yellow"),
                    ),
                    legendgroup=side,
                    hovertemplate=f"<b>{name}</b> [POST-THROW]<br>{side} - {player_input['player_position'][0]}<br>{player_role}<br>Frame: %{{pointNumber}}<br>x=%{{x:.2f}}, y=%{{y:.2f}}<extra></extra>",
                )
            )

        # Add predictions if available
        if pred_df is not None:
            player_pred = pred_df.filter(pl.col("nfl_id") == nfl_id).sort("frame_id")
            if len(player_pred) > 0:
                x_pred = player_pred["x"].to_list()
                y_pred = player_pred["y"].to_list()
                fig.add_trace(
                    go.Scatter(
                        x=x_pred,
                        y=y_pred,
                        mode="lines+markers",
                        name=f"{name} (PREDICTION)",
                        line=dict(color="magenta", width=3, dash="dot"),
                        marker=dict(size=4, color="magenta", opacity=0.7),
                        showlegend=False,
                        hovertemplate=f"{name} [PREDICTION]<br>Frame: %{{pointNumber}}<br>x=%{{x:.2f}}, y=%{{y:.2f}}<extra></extra>",
                    )
                )

    # Add ball landing location
    ball_x = input_df["ball_land_x"][0]
    ball_y = input_df["ball_land_y"][0]
    fig.add_trace(
        go.Scatter(
            x=[ball_x],
            y=[ball_y],
            mode="markers",
            name="Ball Landing",
            marker=dict(
                size=15, color="gold", symbol="star", line=dict(color="black", width=2)
            ),
            hovertemplate=f"Ball Landing<br>x={ball_x:.2f}, y={ball_y:.2f}<extra></extra>",
        )
    )

    return fig


def main():
    st.set_page_config(page_title="NFL Data Bowl 2025", layout="wide")
    st.title("🏈 NFL Data Bowl 2025 - Player Tracking Visualization")

    st.markdown(
        """
    Visualize player movement during pass plays:
    - **Thin dashed lines (faded)**: PRE-THROW movement (before QB releases ball)
    - **Arrows at end of pre-throw**: Player orientation direction
    - **Black square with "QB" label**: Ball thrower position at release
    - **Thick solid lines**: POST-THROW movement (ball in air)
    - **Thick dash-dot lines with yellow outline**: Players to predict (⭐ in legend)
    - **Magenta dotted lines**: PREDICTED trajectories (if loaded)
    - **⭐ Gold star**: Ball landing location
    - **Hover over traces** to see player names and details
    """
    )

    # Sidebar controls
    st.sidebar.header("Controls")

    # Week selection
    week = st.sidebar.selectbox("Select Week", range(1, 19), index=0)

    # Load data
    input_df, output_df = load_data(week)
    games = get_unique_games(input_df)

    # Game selection
    if st.sidebar.button("🎲 Random Game"):
        st.session_state.game_id = random.choice(games)

    game_id = st.sidebar.selectbox(
        "Select Game ID",
        games,
        index=(
            games.index(st.session_state.game_id)
            if "game_id" in st.session_state and st.session_state.game_id in games
            else 0
        ),
    )

    # Play selection
    plays = get_plays_for_game(input_df, game_id)

    if "play_idx" not in st.session_state:
        st.session_state.play_idx = 0

    # Navigation buttons
    col1, col2, col3 = st.sidebar.columns(3)
    if col1.button("⏮️ First"):
        st.session_state.play_idx = 0
    if col2.button("⬅️ Prev"):
        st.session_state.play_idx = max(0, st.session_state.play_idx - 1)
    if col3.button("➡️ Next"):
        st.session_state.play_idx = min(len(plays) - 1, st.session_state.play_idx + 1)

    st.session_state.play_idx = st.sidebar.slider(
        "Play Index", 0, len(plays) - 1, st.session_state.play_idx
    )

    play_id = plays[st.session_state.play_idx]

    # Prediction upload
    st.sidebar.markdown("---")
    pred_file = st.sidebar.file_uploader(
        "Upload Predictions (optional)",
        type=["csv"],
        help="Upload CSV with columns: game_id, play_id, nfl_id, frame_id, x, y",
    )

    # Filter data for selected game and play
    play_input = input_df.filter(
        (pl.col("game_id") == game_id) & (pl.col("play_id") == play_id)
    )
    play_output = output_df.filter(
        (pl.col("game_id") == game_id) & (pl.col("play_id") == play_id)
    )

    # Load predictions if uploaded
    pred_df = None
    if pred_file is not None:
        pred_full = load_predictions(pred_file)
        pred_df = pred_full.filter(
            (pl.col("game_id") == game_id) & (pl.col("play_id") == play_id)
        )

    # Display info
    st.subheader(
        f"Game: {game_id} | Play: {play_id} ({st.session_state.play_idx + 1}/{len(plays)})"
    )

    # Play details
    col1, col2, col3, col4 = st.columns(4)
    play_direction = play_input["play_direction"][0]
    yardline = play_input["absolute_yardline_number"][0]
    num_frames = play_output.shape[0] // play_output["nfl_id"].n_unique()
    ball_land = (
        f"({play_input['ball_land_x'][0]:.1f}, {play_input['ball_land_y'][0]:.1f})"
    )

    col1.metric("Play Direction", play_direction)
    col2.metric("Yardline", yardline)
    col3.metric("Frames (post-throw)", num_frames)
    col4.metric("Ball Landing", ball_land)

    # Create and display plot
    fig = create_field_plot()
    fig = add_traces_to_plot(fig, play_input, play_output, pred_df)
    fig.update_layout(title=f"Player Trajectories - Game {game_id}, Play {play_id}")

    st.plotly_chart(fig, use_container_width=True)

    # Player details
    with st.expander("📊 Player Details"):
        players_info = (
            play_input.select(
                [
                    "player_name",
                    "player_position",
                    "player_side",
                    "player_role",
                    "player_to_predict",
                ]
            )
            .unique()
            .sort("player_side", "player_position")
        )
        st.dataframe(players_info, use_container_width=True)


if __name__ == "__main__":
    main()
