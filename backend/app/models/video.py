"""SQLAlchemy models for video upload, prediction, and frame results."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.models.base import Base

if TYPE_CHECKING:
    pass


class Video(Base):
    """Uploaded video file."""

    __tablename__ = "videos"

    id = Column(String(36), primary_key=True)
    filename = Column(String(255), nullable=False)
    storage_path = Column(String(512), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    duration_sec = Column(Float, nullable=True)
    status = Column(String(32), default="pending")  # pending, processing, done, error
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    prediction = relationship("Prediction", back_populates="video", uselist=False)
    frame_results = relationship("FrameResult", back_populates="video", order_by="FrameResult.frame_index")


class Prediction(Base):
    """Video-level prediction result."""

    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False, unique=True)
    video_score = Column(Float, nullable=False)
    prediction = Column(String(16), nullable=False)  # real, fake
    num_frames = Column(Integer, nullable=True)
    processing_time_sec = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="prediction")
    frame_results = relationship("FrameResult", back_populates="prediction")


class FrameResult(Base):
    """Per-frame prediction and optional heatmap."""

    __tablename__ = "frame_results"

    id = Column(String(36), primary_key=True)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False)
    prediction_id = Column(String(36), ForeignKey("predictions.id"), nullable=False)
    frame_index = Column(Integer, nullable=False)
    frame_score = Column(Float, nullable=False)
    heatmap_url = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("Video", back_populates="frame_results")
    prediction = relationship("Prediction", back_populates="frame_results")
