from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, DateTime, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    smiles = Column(String, unique=True, index=True, nullable=False)
    target = Column(String, index=True)
    mw = Column(Float)
    logp = Column(Float)
    tpsa = Column(Float)
    qed = Column(Float)
    sa_score = Column(Float)
    lipinski_pass = Column(Boolean)
    is_novel = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    binding_affinity = relationship("BindingAffinity", back_populates="candidate", uselist=False, cascade="all, delete-orphan")
    toxicity = relationship("Toxicity", back_populates="candidate", uselist=False, cascade="all, delete-orphan")
    admet = relationship("ADMET", back_populates="candidate", uselist=False, cascade="all, delete-orphan")
    human_anatomy = relationship("HumanAnatomy", back_populates="candidate", uselist=False, cascade="all, delete-orphan")

class BindingAffinity(Base):
    __tablename__ = "binding_affinity"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id", ondelete="CASCADE"), unique=True)
    xgb_pic50 = Column(Float)
    qsvr_pic50 = Column(Float)
    scoring_mode = Column(String)
    latency_s = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="binding_affinity")

class Toxicity(Base):
    __tablename__ = "toxicity"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id", ondelete="CASCADE"), unique=True)
    canonical_smiles = Column(String)
    toxicity_score = Column(Float)
    is_toxic = Column(Boolean)
    alerts_json = Column(JSON)  # stores structural alerts
    created_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="toxicity")

class ADMET(Base):
    """Schema only - functionality to be added in later phase."""
    __tablename__ = "admet"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id", ondelete="CASCADE"), unique=True)
    absorption = Column(Float)
    distribution = Column(Float)
    metabolism = Column(Float)
    excretion = Column(Float)
    overall = Column(Float)
    verdict = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="admet")

class HumanAnatomy(Base):
    """Schema only - functionality for 3D Z-Anatomy mapping to be added in later phase."""
    __tablename__ = "human_anatomy"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id", ondelete="CASCADE"), unique=True)
    disease_target = Column(String, index=True) # E.g., Lung Cancer, Arthritis
    affected_body_parts = Column(JSON)          # List of affected regions/organs
    helped_body_parts = Column(JSON)            # Parts positively impacted
    worsened_body_parts = Column(JSON)          # Side effects / parts negatively impacted
    details = Column(Text)                      # Additional 3D interaction detailing
    created_at = Column(DateTime, default=datetime.utcnow)

    candidate = relationship("Candidate", back_populates="human_anatomy")

class Experiment(Base):
    """Stores complete experiment sessions for persistent access."""
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    pdb_id = Column(String, index=True, nullable=False)
    target_name = Column(String)
    temperature = Column(Float)
    n_candidates = Column(Integer)
    stress_factors = Column(JSON)          # e.g. ["mutation", "thermal"]
    docking_engine = Column(String)
    vqe_optimizer = Column(String)
    vqe_max_iterations = Column(Integer)
    run_admet = Column(Boolean, default=True)
    generation_time_s = Column(Float)
    n_sampled = Column(Integer)
    n_valid = Column(Integer)
    candidates_json = Column(JSON)         # Full candidates array with all scores
    created_at = Column(DateTime, default=datetime.utcnow)
