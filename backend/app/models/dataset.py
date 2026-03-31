"""Dataset SQLAlchemy Model."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, Integer, BigInteger, DateTime, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False)  # csv, parquet, json, xlsx
    num_rows: Mapped[int] = mapped_column(Integer, nullable=True)
    num_columns: Mapped[int] = mapped_column(Integer, nullable=True)
    column_names: Mapped[dict] = mapped_column(JSON, nullable=True)  # list of column names
    column_types: Mapped[dict] = mapped_column(JSON, nullable=True)  # column name -> dtype
    summary_stats: Mapped[dict] = mapped_column(JSON, nullable=True)  # profiling summary
    checksum: Mapped[str] = mapped_column(String(64), nullable=True)  # SHA-256
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    project = relationship("Project", back_populates="datasets")
    jobs = relationship("Job", back_populates="dataset")
