from torch.utils.data import Dataset, DataLoader
import polars as pl


class NFLData(Dataset):
    def __init__(self, inp_df, out_df, feature_cols=None):
        super().__init__()
        self.inp_df, self.out_df = None, None
        self._prepare_data(inp_df, out_df)
        self.game_play_nfl_ids = self.inp_df["game_play_nfl_id"].unique().to_numpy()
        self.feature_cols = feature_cols if feature_cols is not None else ["x", "y"]

    def _prepare_data(self, inp_df, out_df):
        inp_df = inp_df.sort(["game_id", "play_id", "frame_id"])
        out_df = out_df.sort(["game_id", "play_id", "frame_id"])
        self.inp_df = inp_df.with_columns(
            pl.struct(pl.col("game_id"), pl.col("play_id"), pl.col("nfl_id")).map_elements(
                lambda row: f"{row['game_id']}_{row['play_id']}_{row['nfl_id']}"
            ).alias('game_play_nfl_id')
        ).group_by(["game_play_nfl_id"], maintain_order=True).agg(
            pl.col("x"),
            pl.col("y"),
            pl.col("s"),
            pl.col("a"),
            pl.col("dir"),
            pl.col("o"),
            pl.col("frame_id").max() - pl.col("frame_id"),
            pl.col("player_to_predict").first(),
            pl.col("num_frames_output").first(),
            pl.col("ball_land_x").first(),
            pl.col("ball_land_y").first(),
            pl.col("player_role").first(),
            pl.col("player_name").first(),
            pl.col("player_height").first(),
            pl.col("player_weight").first(),
            pl.col("player_birth_date").first(),
            pl.col("player_position").first(),
            pl.col("player_side").first(),
        )

        self.out_df = out_df.with_columns(
            pl.struct(pl.col("game_id"), pl.col("play_id"), pl.col("nfl_id")).map_elements(
                lambda row: f"{row['game_id']}_{row['play_id']}_{row['nfl_id']}"
            ).alias('game_play_nfl_id')
        ).group_by(["game_play_nfl_id"], maintain_order=True).agg(
            pl.col("x"),
            pl.col("y"),
            pl.col("frame_id"),
        )

    def __len__(self):
        return len(self.game_play_nfl_ids)

    def __getitem__(self, idx):
        game_play_id = self.game_play_nfl_ids[idx]
        inp_data = self.inp_df.filter(
            pl.col("game_play_nfl_id") == game_play_id
        ).select(*self.feature_cols).to_numpy()

        out_data = self.out_df.filter(
            pl.col("game_play_nfl_id") == game_play_id
        ).select(*self.feature_cols).to_numpy()
        return inp_data, out_data


# Example:

# B, L, F -> AutoRegression 
# B, L, 2 -