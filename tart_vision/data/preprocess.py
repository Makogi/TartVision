import pandas as pd

def preprocess_features(df):
    print("正在计算积热时间特征 (Time at 100°C)...")
    df = df.copy()

    if 'time' in df.columns:
        df['time_timestamp'] = pd.to_datetime(df['time'])
    else:
        df['time_timestamp'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(df.index, unit='s')

    df = df.sort_values(by=['experiment_id', 'tart_id', 'time_timestamp'])

    def calculate_plateau_time(group):
        is_plateau = group['temperature'] >= 98.0
        group['time_at_100'] = 0.0

        if is_plateau.any():
            start_time = group.loc[is_plateau, 'time_timestamp'].min()
            mask = group['time_timestamp'] >= start_time
            time_diff = (group.loc[mask, 'time_timestamp'] - start_time).dt.total_seconds() / 60.0
            group.loc[mask, 'time_at_100'] = time_diff
        return group

    df = df.groupby(['experiment_id', 'tart_id']).apply(calculate_plateau_time).reset_index(drop=True)
    df['time_at_100'] = df['time_at_100'] / 30.0
    df['time_at_100'] = df['time_at_100'].fillna(0.0)

    print("特征计算完成。")
    return df