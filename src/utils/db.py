import sqlite3
import pandas as pd
from contextlib import contextmanager
from src.utils.config import DB_PATH

DDL = """
CREATE TABLE IF NOT EXISTS torvik_ratings (
    year INTEGER,
    team TEXT,
    conf TEXT,
    adj_o REAL, adj_d REAL, adj_t REAL, barthag REAL,
    efg_o REAL, efg_d REAL,
    to_rate_o REAL, to_rate_d REAL,
    or_rate_o REAL, or_rate_d REAL,
    ft_rate_o REAL, ft_rate_d REAL,
    three_pt_rate_o REAL, three_pt_rate_d REAL,
    three_pt_pct_o REAL, three_pt_pct_d REAL,
    two_pt_pct_o REAL, two_pt_pct_d REAL,
    sos REAL, seed INTEGER,
    updated_at TIMESTAMP,
    PRIMARY KEY (year, team)
);

CREATE TABLE IF NOT EXISTS torvik_ratings_snapshot (
    year INTEGER,
    team TEXT,
    as_of_date TEXT,
    conf TEXT,
    adj_o REAL, adj_d REAL, adj_t REAL, barthag REAL,
    efg_o REAL, efg_d REAL,
    to_rate_o REAL, to_rate_d REAL,
    or_rate_o REAL, or_rate_d REAL,
    ft_rate_o REAL, ft_rate_d REAL,
    three_pt_rate_o REAL, three_pt_rate_d REAL,
    three_pt_pct_o REAL, three_pt_pct_d REAL,
    two_pt_pct_o REAL, two_pt_pct_d REAL,
    sos REAL, seed INTEGER,
    updated_at TIMESTAMP,
    PRIMARY KEY (year, team, as_of_date)
);

CREATE TABLE IF NOT EXISTS torvik_games (
    year INTEGER,
    game_date TEXT,
    team TEXT,
    opponent TEXT,
    location TEXT,
    team_score INTEGER,
    opp_score INTEGER,
    is_tournament BOOLEAN,
    tournament_round TEXT,
    PRIMARY KEY (year, game_date, team, opponent)
);

CREATE TABLE IF NOT EXISTS odds_history (
    game_id TEXT,
    pull_timestamp TIMESTAMP,
    home_team TEXT,
    away_team TEXT,
    commence_time TIMESTAMP,
    spread_home REAL,
    spread_away REAL,
    total_line REAL,
    bookmaker TEXT,
    is_opening BOOLEAN,
    PRIMARY KEY (game_id, pull_timestamp, bookmaker)
);

CREATE TABLE IF NOT EXISTS historical_results (
    year INTEGER,
    round_number INTEGER,
    round_name TEXT,
    game_date TEXT,
    team1 TEXT,
    team2 TEXT,
    score1 INTEGER,
    score2 INTEGER,
    winner TEXT,
    margin INTEGER,
    total_points INTEGER,
    seed1 INTEGER,
    seed2 INTEGER,
    PRIMARY KEY (year, game_date, team1, team2)
);

CREATE TABLE IF NOT EXISTS historical_lines (
    year INTEGER,
    game_date TEXT,
    team1 TEXT,
    team2 TEXT,
    spread_favorite TEXT,
    spread_line REAL,
    total_line REAL,
    open_spread REAL,
    open_total REAL,
    ats_result TEXT,
    ou_result TEXT,
    source TEXT,
    PRIMARY KEY (year, game_date, team1, team2)
);

CREATE TABLE IF NOT EXISTS mm_training_data (
    year INTEGER,
    game_date TEXT,
    team_a TEXT,
    team_b TEXT,
    seed_a INTEGER,
    seed_b INTEGER,
    round_number INTEGER,
    adj_o_diff REAL, adj_d_diff REAL, avg_tempo REAL,
    tempo_diff REAL, barthag_diff REAL,
    efg_o_diff REAL, efg_d_diff REAL,
    to_rate_diff REAL, or_rate_diff REAL, ft_rate_diff REAL,
    three_pt_rate_diff REAL, three_pt_pct_diff REAL, two_pt_pct_diff REAL,
    sos_diff REAL, conf_power_a REAL, conf_power_b REAL,
    days_rest_diff INTEGER,
    actual_margin REAL,
    actual_total REAL,
    market_spread REAL,
    market_total REAL,
    PRIMARY KEY (year, game_date, team_a, team_b)
);

CREATE TABLE IF NOT EXISTS tournament_bracket (
    year INTEGER,
    region TEXT,
    seed INTEGER,
    team TEXT,
    fetched_at TIMESTAMP,
    PRIMARY KEY (year, region, seed)
);

CREATE TABLE IF NOT EXISTS nit_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    year         INTEGER NOT NULL,
    game_date    TEXT NOT NULL,
    team1        TEXT NOT NULL,
    team2        TEXT NOT NULL,
    score1       INTEGER,
    score2       INTEGER,
    round_number INTEGER,
    round_name   TEXT,
    UNIQUE(year, game_date, team1, team2)
);

CREATE TABLE IF NOT EXISTS nit_lines (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    year            INTEGER NOT NULL,
    game_date       TEXT NOT NULL,
    team1           TEXT NOT NULL,
    team2           TEXT NOT NULL,
    spread_line     REAL,
    total_line      REAL,
    spread_favorite TEXT,
    source          TEXT,
    UNIQUE(year, game_date, team1, team2)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    year INTEGER,
    game_date TEXT,
    team_a TEXT,
    team_b TEXT,
    round_number INTEGER,
    model_spread REAL,
    model_total REAL,
    market_spread REAL,
    market_total REAL,
    spread_edge REAL,
    total_edge REAL,
    win_prob_a REAL,
    actual_margin REAL,
    actual_total REAL,
    spread_result TEXT,
    total_result TEXT
);

CREATE TABLE IF NOT EXISTS halftime_scores (
    year           INTEGER,
    game_date      TEXT,
    team1          TEXT,
    team2          TEXT,
    espn_game_id   TEXT,
    h1_score1      INTEGER,
    h1_score2      INTEGER,
    h1_margin      REAL,
    h1_combined    INTEGER,
    h1_efg1        REAL,
    h1_efg2        REAL,
    h1_orb1        INTEGER,
    h1_orb2        INTEGER,
    h1_to1         INTEGER,
    h1_to2         INTEGER,
    source         TEXT,
    PRIMARY KEY (year, game_date, team1, team2)
);

CREATE TABLE IF NOT EXISTS live_game_snapshots (
    snapshot_id    TEXT PRIMARY KEY,
    espn_game_id   TEXT,
    game_date      TEXT,
    team1          TEXT,
    team2          TEXT,
    snapshot_ts    TIMESTAMP,
    period         INTEGER,
    clock_secs     INTEGER,
    score1         INTEGER,
    score2         INTEGER,
    current_margin REAL,
    time_elapsed   REAL,
    time_remaining REAL,
    efg_pct1       REAL,
    efg_pct2       REAL,
    orb1           INTEGER,
    orb2           INTEGER,
    to1            INTEGER,
    to2            INTEGER,
    game_status    TEXT,
    live_spread    REAL,
    live_total     REAL,
    pregame_spread REAL,
    projected_total REAL
);

CREATE TABLE IF NOT EXISTS live_predictions (
    prediction_id     TEXT PRIMARY KEY,
    snapshot_id       TEXT,
    espn_game_id      TEXT,
    game_date         TEXT,
    team1             TEXT,
    team2             TEXT,
    time_elapsed      REAL,
    time_remaining    REAL,
    game_status       TEXT,
    projected_margin  REAL,
    live_market_spread REAL,
    edge              REAL,
    tier              TEXT,
    kelly_pct         REAL,
    bet_team          TEXT,
    component_time_weight   REAL,
    component_efg_adj       REAL,
    component_orb_adj       REAL,
    component_to_adj        REAL,
    created_at        TIMESTAMP
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_conn():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db_conn() as conn:
        conn.executescript(DDL)
        # Migration: add espn_game_id to historical_results if not present
        try:
            conn.execute("ALTER TABLE historical_results ADD COLUMN espn_game_id TEXT")
            conn.commit()
        except Exception:
            pass  # Column already exists
    print(f"Database initialized at {DB_PATH}")


def execute_query(sql: str, params=None) -> list:
    with db_conn() as conn:
        cur = conn.execute(sql, params or [])
        return [dict(row) for row in cur.fetchall()]


def query_df(sql: str, params=None) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def upsert_df(df: pd.DataFrame, table: str, if_exists: str = "replace"):
    with get_connection() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False, method="multi")


def get_table_count(table: str) -> int:
    rows = execute_query(f"SELECT COUNT(*) as cnt FROM {table}")
    return rows[0]["cnt"] if rows else 0
