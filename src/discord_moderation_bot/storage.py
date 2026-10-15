"""Small SQLite store for warning counts; no message text is persisted."""

import sqlite3
from pathlib import Path


class WarningStore:
    def __init__(self, database_path: Path):
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS warnings (
                    guild_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    warning_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (guild_id, user_id)
                )"""
            )

    def add_warning(self, guild_id: int, user_id: int) -> int:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """INSERT INTO warnings (guild_id, user_id, warning_count)
                   VALUES (?, ?, 1)
                   ON CONFLICT(guild_id, user_id)
                   DO UPDATE SET warning_count = warning_count + 1""",
                (guild_id, user_id),
            )
            row = connection.execute(
                "SELECT warning_count FROM warnings WHERE guild_id = ? AND user_id = ?",
                (guild_id, user_id),
            ).fetchone()
        return int(row[0])

    def clear_warnings(self, guild_id: int, user_id: int) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "DELETE FROM warnings WHERE guild_id = ? AND user_id = ?",
                (guild_id, user_id),
            )
