from fastapi import FastAPI, HTTPException, Depends, status
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# Database Configuration
SQLALCHEMY_DATABASE_URL = "sqlite:///./db.sqlite3"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# SQLAlchemy Model
class CustomOntology(Base):
    __tablename__ = "custom_accounts_customontologyupload"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    ontology_type = Column(String(150), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    edit_uid = Column(Integer, nullable=False)


# Pydantic Models
class OntologyBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    content: str
    ontology_type: str = Field(..., min_length=1, max_length=150)
    edit_uid: int


class OntologyCreate(OntologyBase):
    pass


class OntologyUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    content: Optional[str] = None
    ontology_type: Optional[str] = Field(None, min_length=1, max_length=150)
    edit_uid: Optional[int] = None


class OntologyResponse(OntologyBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Database Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# FastAPI Application
app = FastAPI(title="Custom Ontology API")


@app.on_event("startup")
async def startup():
    Base.metadata.create_all(bind=engine)


@app.post("/ontologies/", response_model=OntologyResponse, status_code=status.HTTP_201_CREATED)
def create_ontology(ontology: OntologyCreate, db: Session = Depends(get_db)):
    """
    Create a new ontology entry.
    """
    db_ontology = CustomOntology(
        name=ontology.name,
        content=ontology.content,
        ontology_type=ontology.ontology_type,
        edit_uid=ontology.edit_uid,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    try:
        db.add(db_ontology)
        db.commit()
        db.refresh(db_ontology)
        return db_ontology
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ontologies/", response_model=List[OntologyResponse])
def list_ontologies(
        skip: int = 0,
        limit: int = 100,
        db: Session = Depends(get_db)
):
    """
    Retrieve a list of ontologies with pagination.
    """
    ontologies = db.query(CustomOntology).offset(skip).limit(limit).all()
    return ontologies


@app.get("/ontologies/{ontology_id}", response_model=OntologyResponse)
def get_ontology(ontology_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a specific ontology by ID.
    """
    ontology = db.query(CustomOntology).filter(CustomOntology.id == ontology_id).first()
    if not ontology:
        raise HTTPException(status_code=404, detail="Ontology not found")
    return ontology


@app.put("/ontologies/{ontology_id}", response_model=OntologyResponse)
def update_ontology(
        ontology_id: int,
        ontology_update: OntologyUpdate,
        db: Session = Depends(get_db)
):
    """
    Update an existing ontology.
    """
    db_ontology = db.query(CustomOntology).filter(CustomOntology.id == ontology_id).first()
    if not db_ontology:
        raise HTTPException(status_code=404, detail="Ontology not found")

    update_data = ontology_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_ontology, field, value)

    db_ontology.updated_at = datetime.utcnow()

    try:
        db.commit()
        db.refresh(db_ontology)
        return db_ontology
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/ontologies/{ontology_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_ontology(ontology_id: int, db: Session = Depends(get_db)):
    """
    Delete an ontology.
    """
    ontology = db.query(CustomOntology).filter(CustomOntology.id == ontology_id).first()
    if not ontology:
        raise HTTPException(status_code=404, detail="Ontology not found")

    try:
        db.delete(ontology)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)