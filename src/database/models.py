from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[Optional[str]] = mapped_column(String(256))
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    incidents: Mapped[list["Incident"]] = relationship(back_populates="camera")


class Criminal(Base):
    __tablename__ = "criminals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    age: Mapped[Optional[int]] = mapped_column(Integer)
    gender: Mapped[Optional[str]] = mapped_column(String(16))
    crime_type: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    threat_level: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    photo_path: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    embeddings: Mapped[list["Embedding"]] = relationship(back_populates="criminal", cascade="all, delete-orphan")
    incidents: Mapped[list["IncidentMatch"]] = relationship(back_populates="criminal")


class Embedding(Base):
    __tablename__ = "embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    criminal_id: Mapped[int] = mapped_column(ForeignKey("criminals.id", ondelete="CASCADE"), index=True)
    vector: Mapped[bytes] = mapped_column("vector", Text, nullable=False)  # store as JSON string of list or base64; persisted FAISS used for realtime
    dim: Mapped[int] = mapped_column(Integer, default=512)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    criminal: Mapped[Criminal] = relationship(back_populates="embeddings")


class Incident(Base):
    __tablename__ = "incidents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    image_path: Mapped[Optional[str]] = mapped_column(Text)  # optional snapshot path
    face_count: Mapped[int] = mapped_column(Integer, default=0)

    camera: Mapped[Camera] = relationship(back_populates="incidents")
    matches: Mapped[list["IncidentMatch"]] = relationship(back_populates="incident", cascade="all, delete-orphan")


class IncidentMatch(Base):
    __tablename__ = "incident_matches"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"), index=True)
    criminal_id: Mapped[int] = mapped_column(ForeignKey("criminals.id"), index=True)
    similarity: Mapped[float] = mapped_column(Float, index=True)
    bbox_x1: Mapped[int] = mapped_column(Integer)
    bbox_y1: Mapped[int] = mapped_column(Integer)
    bbox_x2: Mapped[int] = mapped_column(Integer)
    bbox_y2: Mapped[int] = mapped_column(Integer)

    incident: Mapped[Incident] = relationship(back_populates="matches")
    criminal: Mapped[Criminal] = relationship(back_populates="incidents")
