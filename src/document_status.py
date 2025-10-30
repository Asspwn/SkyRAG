from typing import Dict, Any, List
from datetime import datetime
import threading


class DocumentStatusTracker:
    #Statuses: indexing, completed, failed
    
    def __init__(self):
        self._statuses: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_document(self, document_id: str, user_id: str, filename: str) -> None:
        """Create a new document with 'indexing' status"""
        with self._lock:
            self._statuses[document_id] = {
                "document_id": document_id,
                "user_id": user_id,
                "filename": filename,
                "status": "indexing",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "error": None
            }

    def update_status(self, document_id: str, status: str, error: str = None) -> None:
        """Update document status (indexing, completed, failed)"""
        with self._lock:
            if document_id in self._statuses:
                self._statuses[document_id]["status"] = status
                self._statuses[document_id]["updated_at"] = datetime.utcnow().isoformat()
                if error:
                    self._statuses[document_id]["error"] = error

    def get_status(self, document_id: str) -> Dict[str, Any]:
        """Get status of a document"""
        with self._lock:
            return self._statuses.get(document_id)

    def get_user_documents(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a user"""
        with self._lock:
            return [
                doc for doc in self._statuses.values()
                if doc["user_id"] == user_id
            ]

    def delete_document(self, document_id: str) -> bool:
        """Delete a document from status tracker"""
        with self._lock:
            if document_id in self._statuses:
                del self._statuses[document_id]
                return True
            return False

status_tracker = DocumentStatusTracker()
