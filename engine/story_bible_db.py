from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    genre_pack: Mapped[str] = mapped_column(String(80), default="thriller_scifi")
    status: Mapped[str] = mapped_column(String(40), default="draft")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    characters: Mapped[list[Character]] = relationship(back_populates="project", cascade="all, delete-orphan")
    plot_threads: Mapped[list[PlotThread]] = relationship(back_populates="project", cascade="all, delete-orphan")
    world_facts: Mapped[list[WorldFact]] = relationship(back_populates="project", cascade="all, delete-orphan")


class Character(Base):
    __tablename__ = "characters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(80), default="supporting")
    core_wound: Mapped[str] = mapped_column(Text, default="")
    flawed_belief: Mapped[str] = mapped_column(Text, default="")
    voice_style: Mapped[str] = mapped_column(Text, default="")
    story_function: Mapped[str] = mapped_column(Text, default="")
    arc_state: Mapped[str] = mapped_column(Text, default="")

    project: Mapped[Project] = relationship(back_populates="characters")


class PlotThread(Base):
    __tablename__ = "plot_threads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    status: Mapped[str] = mapped_column(String(40), default="open")
    introduced_chapter: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    resolved_chapter: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    description: Mapped[str] = mapped_column(Text, default="")

    project: Mapped[Project] = relationship(back_populates="plot_threads")


class WorldFact(Base):
    __tablename__ = "world_facts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    key: Mapped[str] = mapped_column(String(160), index=True)
    value: Mapped[str] = mapped_column(Text)
    scope: Mapped[str] = mapped_column(String(80), default="global")
    genre_specific: Mapped[dict] = mapped_column(JSON, default=dict)

    project: Mapped[Project] = relationship(back_populates="world_facts")


class Theme(Base):
    __tablename__ = "themes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    primary_statement: Mapped[str] = mapped_column(Text)
    secondary_threads: Mapped[dict] = mapped_column(JSON, default=dict)


class TimelineEntry(Base):
    __tablename__ = "timeline_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    chapter_num: Mapped[int] = mapped_column(Integer)
    scene_num: Mapped[int] = mapped_column(Integer)
    event: Mapped[str] = mapped_column(Text)
    location: Mapped[str] = mapped_column(String(160), default="")
    characters_present: Mapped[dict] = mapped_column(JSON, default=dict)


class CheckpointState(Base):
    __tablename__ = "checkpoint_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    fsm_state: Mapped[str] = mapped_column(String(80))
    chapter_num: Mapped[int] = mapped_column(Integer)
    scene_num: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def make_engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}", future=True)


def make_session_factory(db_path: str):
    engine = make_engine(db_path)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
