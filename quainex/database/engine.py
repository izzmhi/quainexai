"""Database engine and session management.

Purpose:
    Own the connection to Quainex's store, and hand out transactional sessions.

Why SQLite, and why async:
    A personal AI OS has exactly one user and no operations team. SQLite needs no
    server, no credentials and no backups beyond copying a file — the right shape
    for something that lives on a laptop. The async driver keeps a slow write off
    the event loop that is simultaneously handling a WebSocket and a voice turn.

    The engine is created through SQLAlchemy's URL layer rather than a hardcoded
    connection string, so the PostgreSQL migration the roadmap anticipates is a
    configuration change plus a driver, not a rewrite.

Architecture:
    Database.create(settings)      build engine, ensure schema
        -> Database.session()      async context manager, commits or rolls back
        -> SqlAlchemyMemoryStore   the only consumer
        -> Database.aclose()       dispose the pool on shutdown

Dependencies:
    sqlalchemy, aiosqlite

Future improvements:
    * Alembic migrations. ``create_all`` is fine while the schema is additive and
      the only user is the developer; it cannot alter an existing column.
    * WAL mode and a busy timeout once background jobs write concurrently.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from quainex.core.logging import get_logger
from quainex.models.memory import Base

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncEngine

    from quainex.config.settings import Settings

_log = get_logger(__name__)


class Database:
    """Owns the engine and hands out sessions."""

    def __init__(self, engine: AsyncEngine) -> None:
        """Construct around an existing engine.

        Args:
            engine: The async engine to use.
        """
        self._engine = engine
        # expire_on_commit=False keeps loaded objects usable after the session
        # closes; otherwise every attribute read after a commit would trigger a
        # lazy refresh against a session that no longer exists.
        self._sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    @classmethod
    def create(cls, settings: Settings) -> Database:
        """Build a database from configuration.

        Args:
            settings: Configuration supplying the database URL.

        Returns:
            A ready database, whose schema has not yet been created.
        """
        url = settings.database_url
        if url.startswith("sqlite"):
            settings.database_path.parent.mkdir(parents=True, exist_ok=True)

        engine = create_async_engine(url, echo=False, future=True)
        _log.info("database_engine_created", dialect=url.split(":", 1)[0])
        return cls(engine)

    async def create_schema(self) -> None:
        """Create any missing tables.

        Additive only: this creates tables that do not exist and never alters
        ones that do. Changing an existing column will need Alembic.
        """
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        _log.info("database_schema_ready", tables=len(Base.metadata.tables))

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a transactional session.

        Commits when the block completes, rolls back if it raises. Callers never
        manage the transaction themselves, so a half-written memory cannot be
        left behind by an exception mid-write.

        Yields:
            An open session.
        """
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def aclose(self) -> None:
        """Dispose of the connection pool."""
        await self._engine.dispose()
        _log.info("database_engine_closed")
