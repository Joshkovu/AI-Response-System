import json
import sqlite3
from datetime import datetime
from typing import Optional
import os
import threading


class DocumentStore:
    """
    SQLite-backed persistent storage for indexed documents and their vectors.
    
    Supports:
    - Store/retrieve document metadata (name, category, upload_date, page_count)
    - Store/retrieve vectors associated with documents
    - Query and list all indexed documents
    - Delete documents and their associated vectors
    """

    def __init__(self, db_path: str = "document_index.db"):
        """Initialize the document store and create schema if needed."""
        self.db_path = db_path
        self.conn = None
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        """Create SQLite tables if they don't exist."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()

        # Documents table: metadata about indexed PDFs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL UNIQUE,
                category TEXT,
                page_count INTEGER,
                upload_date TEXT,
                embedding_model TEXT DEFAULT 'gemini-embedding-001',
                chunk_count INTEGER DEFAULT 0
            )
        """)

        # Chunks table: individual indexed text chunks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                vector BLOB NOT NULL,
                created_at TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
            )
        """)

        # Index on doc_id for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id)
        """)

        self.conn.commit()

    def add_document(
        self,
        file_name: str,
        chunks: list[str],
        vectors: list[list[float]],
        page_count: int = 0,
        category: str = "general",
        embedding_model: str = "gemini-embedding-001",
    ) -> int:
        """
        Store a document and its indexed chunks/vectors.
        
        Args:
            file_name: Original PDF filename
            chunks: List of text chunks
            vectors: List of embedding vectors (same length as chunks)
            page_count: Number of pages in the PDF
            category: Document category/tag
            embedding_model: Model used to generate embeddings
            
        Returns:
            doc_id of the newly inserted document
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        with self._lock:
            cursor = self.conn.cursor()
            now = datetime.utcnow().isoformat()

            # Insert document metadata
            cursor.execute(
                """
                INSERT INTO documents (file_name, category, page_count, upload_date, embedding_model, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (file_name, category, page_count, now, embedding_model, len(chunks)),
            )
            doc_id = cursor.lastrowid

            # Insert chunks and vectors
            for chunk_idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
                vector_bytes = self._serialize_vector(vector)
                cursor.execute(
                    """
                    INSERT INTO chunks (doc_id, chunk_index, content, vector, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (doc_id, chunk_idx, chunk, vector_bytes, now),
                )

            self.conn.commit()
            return doc_id

    def list_documents(self) -> list[dict]:
        """Retrieve all indexed documents with metadata."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT doc_id, file_name, category, page_count, upload_date, chunk_count
                FROM documents
                ORDER BY upload_date DESC
            """)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_document(self, doc_id: int) -> Optional[dict]:
        """Retrieve a single document's metadata."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT doc_id, file_name, category, page_count, upload_date, chunk_count, embedding_model
                FROM documents
                WHERE doc_id = ?
            """, (doc_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_chunks_for_document(self, doc_id: int) -> list[dict]:
        """Retrieve all chunks for a specific document."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT chunk_id, chunk_index, content, vector, created_at
                FROM chunks
                WHERE doc_id = ?
                ORDER BY chunk_index
            """, (doc_id,))
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "chunk_id": row["chunk_id"],
                    "doc_id": doc_id,
                    "chunk_index": row["chunk_index"],
                    "content": row["content"],
                    "vector": self._deserialize_vector(row["vector"]),
                    "created_at": row["created_at"],
                })
            return results

    def load_all_chunks(self) -> tuple[list[list[float]], list[dict]]:
        """
        Load all vectors and their associated metadata for rebuilding the in-memory index.
        
        Returns:
            (vectors, documents) where documents includes doc_id, chunk_id, content, etc.
        """
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT c.chunk_id, c.doc_id, c.chunk_index, c.content, c.vector, d.file_name, d.page_count
                FROM chunks c
                JOIN documents d ON c.doc_id = d.doc_id
                ORDER BY c.doc_id, c.chunk_index
            """)
            rows = cursor.fetchall()

            vectors = []
            documents = []

            for row in rows:
                vector = self._deserialize_vector(row["vector"])
                vectors.append(vector)
                documents.append({
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "file_name": row["file_name"],
                    "content": row["content"],
                    "page_count": row["page_count"],
                })

            return vectors, documents

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document and all its chunks."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            self.conn.commit()
            return cursor.rowcount > 0

    def document_exists(self, file_name: str) -> bool:
        """Check if a document with this filename is already indexed."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM documents WHERE file_name = ?", (file_name,))
            return cursor.fetchone() is not None

    def _serialize_vector(self, vector: list[float]) -> bytes:
        """Serialize a vector to JSON bytes for storage."""
        return json.dumps(vector).encode("utf-8")

    def _deserialize_vector(self, vector_bytes: bytes) -> list[float]:
        """Deserialize a vector from JSON bytes."""
        return json.loads(vector_bytes.decode("utf-8"))

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self.conn:
                self.conn.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
