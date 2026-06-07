import pandas as pd


def clean_data(df: pd.DataFrame, remove_duplicates: bool = False) -> pd.DataFrame:
    print(f"Initial shape: {df.shape}")

    df = df.copy()

    # Ensure datetime columns are valid
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce")

    # Remove rows with invalid timestamps
    df = df.dropna(subset=["started_at", "ended_at"])
    print(f"Shape after removing rows with invalid timestamps: {df.shape}")

    # Remove rows with missing coordinates
    df = df.dropna(subset=["start_lat", "start_lng", "end_lat", "end_lng"])
    print(f"Shape after removing rows with missing coordinates: {df.shape}")

    # Keep only valid coordinate ranges
    df = df[
        (df["start_lat"].between(-90, 90))
        & (df["start_lng"].between(-180, 180))
        & (df["end_lat"].between(-90, 90))
        & (df["end_lng"].between(-180, 180))
    ]
    print(f"Shape after keeping only valid coordinate ranges: {df.shape}")

    if remove_duplicates:
        df = df.drop_duplicates()
        print(f"Shape after removing duplicates: {df.shape}")
    else:
        print("Duplicate removal skipped.")

    print(f"Final cleaned shape: {df.shape}")
    return df
