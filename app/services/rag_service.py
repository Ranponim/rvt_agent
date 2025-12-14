import os
import glob
import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGService:
    """
    RAG (Retrieval-Augmented Generation) Service
    - Manages Vector DB (ChromaDB)
    - Embeds documents using local model
    - Retrieves relevant documents
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.embedding_model_name = "jhgan/ko-sroberta-multitask"
        self.chroma_dir = settings.CHROMA_DB_DIR
        self.collection_name = "peg_knowledge_base"
        
        # 1. Initialize Embedding Model (Lazy Loading)
        self.model = None 

        # 2. Initialize ChromaDB Client
        if not os.path.exists(self.chroma_dir):
            os.makedirs(self.chroma_dir)
            
        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        
        # Get or Create Collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"} # Cosine Similarity
        )
        
        self.kb_dir = os.path.join(settings.KNOWLEDGE_BASE_DIR, "peg_docs")
        self._initialized = True

    def _load_model(self):
        """Load embedding model only when needed to save startup time."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}...")
            self.model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded.")

    def initialize_knowledge_base(self, force_reload: bool = False):
        """
        Loads markdown files from KB directory, chunks them, and indexes into ChromaDB.
        """
        # Check if DB is empty or reload is forced
        if self.collection.count() > 0 and not force_reload:
            logger.info(f"Vector DB already contains {self.collection.count()} items. Skipping initialization.")
            return

        self._load_model()
        
        logger.info("Initializing Knowledge Base from MD files...")
        
        # 1. Load Documents
        docs = []
        md_files = glob.glob(os.path.join(self.kb_dir, "*.md"))
        
        if not md_files:
            logger.warning(f"No markdown files found in {self.kb_dir}")
            return

        # Headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        ids = []
        documents = []
        metadatas = []

        for file_path in md_files:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Split by Header
            header_splits = markdown_splitter.split_text(text)
            
            # Additional Split by Character
            final_splits = text_splitter.split_documents(header_splits)
            
            for idx, split in enumerate(final_splits):
                # Construct Metadata
                source = os.path.basename(file_path)
                meta = split.metadata.copy()
                meta["source"] = source
                
                # Content
                content = split.page_content
                
                # ID
                doc_id = f"{source}_{idx}"
                
                ids.append(doc_id)
                documents.append(content)
                metadatas.append(meta)

        if not documents:
            logger.warning("No content extracted from documents.")
            return

        # 2. Embed Documents
        logger.info(f"Embedding {len(documents)} chunks...")
        embeddings = self.model.encode(documents).tolist()
        
        # 3. Add to Chroma
        # Clear existing if force_reload
        if force_reload and self.collection.count() > 0:
             self.client.delete_collection(self.collection_name)
             self.collection = self.client.create_collection(
                name=self.collection_name, 
                metadata={"hnsw:space": "cosine"}
            )

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            
        logger.info(f"Successfully indexed {len(ids)} chunks to Vector DB.")

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        Search for relevant documents.
        Returns a list of document contents.
        """
        self._load_model()
        
        query_embedding = self.model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        
        # results['documents'] is a list of list of strings
        if results and results['documents']:
            return results['documents'][0]
        
        return []

# Singleton Instance
rag_service = RAGService()
