"""
repositories/sql/document_repo.py
-----------------------------------
SQLAlchemy implementation of DocumentRepoProtocol.
Centralises document lookup, creation, and subject-association queries.
"""

from typing import List, Optional
from core.database import SessionLocal, Document as DBDocument, SubjectDocumentAssociation


def _doc_to_dict(doc: DBDocument) -> dict:
    return {
        "id": doc.id,
        "filename": doc.filename,
        "title": doc.title,
        "content_hash": doc.content_hash,
        "source_type": doc.source_type,
        "source_url": doc.source_url,
        "created_at": doc.created_at,
        "relevance_rate": doc.relevance_rate,
        "yield_rate": doc.yield_rate,
        "faithfulness_score": doc.faithfulness_score,
    }


class DocumentRepo:
    """Concrete SQL implementation of DocumentRepoProtocol."""

    def get_by_content_hash(self, content_hash: str) -> Optional[dict]:
        with SessionLocal() as db:
            doc = db.query(DBDocument).filter(DBDocument.content_hash == content_hash).first()
            return _doc_to_dict(doc) if doc else None

    def create(
        self,
        doc_id: str,
        filename: str,
        title: str,
        content_hash: str,
        source_type: str = "pdf",
        source_url: Optional[str] = None,
    ) -> dict:
        with SessionLocal() as db:
            doc = DBDocument(
                id=doc_id,
                filename=filename,
                title=title,
                content_hash=content_hash,
                source_type=source_type,
                source_url=source_url,
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            return _doc_to_dict(doc)

    def attach_to_subject(self, doc_id: str, subject_id: int) -> None:
        with SessionLocal() as db:
            existing = db.query(SubjectDocumentAssociation).filter(
                SubjectDocumentAssociation.subject_id == subject_id,
                SubjectDocumentAssociation.document_id == doc_id,
            ).first()
            if not existing:
                assoc = SubjectDocumentAssociation(subject_id=subject_id, document_id=doc_id)
                db.add(assoc)
                db.commit()

    def get_attached_to_subject(self, subject_id: int) -> List[dict]:
        with SessionLocal() as db:
            assocs = db.query(SubjectDocumentAssociation).filter(
                SubjectDocumentAssociation.subject_id == subject_id
            ).all()
            doc_ids = [a.document_id for a in assocs]
            if not doc_ids:
                return []
            docs = db.query(DBDocument).filter(DBDocument.id.in_(doc_ids)).all()
            return [_doc_to_dict(d) for d in docs]

    def get_all(self) -> List[dict]:
        with SessionLocal() as db:
            docs = db.query(DBDocument).order_by(DBDocument.created_at.desc()).all()
            return [_doc_to_dict(d) for d in docs]

    def delete(self, doc_id: str) -> None:
        """Delete a document record (ORM cascade handles chunks/topics/flashcards)."""
        with SessionLocal() as db:
            doc = db.query(DBDocument).filter(DBDocument.id == doc_id).first()
            if doc:
                db.delete(doc)
                db.commit()

    def detach_from_subject(self, doc_id: str, subject_id: int) -> None:
        with SessionLocal() as db:
            assoc = db.query(SubjectDocumentAssociation).filter(
                SubjectDocumentAssociation.subject_id == subject_id,
                SubjectDocumentAssociation.document_id == doc_id,
            ).first()
            if assoc:
                db.delete(assoc)
                db.commit()

    def get_not_attached_to_subject(self, subject_id: int) -> List[dict]:
        """All documents that are NOT currently attached to the given subject."""
        with SessionLocal() as db:
            attached_ids = [
                a.document_id
                for a in db.query(SubjectDocumentAssociation).filter(
                    SubjectDocumentAssociation.subject_id == subject_id
                ).all()
            ]
            q = db.query(DBDocument)
            if attached_ids:
                q = q.filter(~DBDocument.id.in_(attached_ids))
            return [_doc_to_dict(d) for d in q.order_by(DBDocument.created_at.desc()).all()]
