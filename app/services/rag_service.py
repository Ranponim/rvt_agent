import os
import glob
import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from app.core.config import settings

# 로거 설정
logger = logging.getLogger(__name__)

class RAGService:
    """
    RAG (Retrieval-Augmented Generation) 서비스
    
    - Vector DB (ChromaDB) 관리 및 검색 담당
    - 로컬 임베딩 모델(SentenceTransformer)을 사용하여 문서 임베딩 수행
    - 기술 문서를 청크(Chunk) 단위로 분할하여 인덱싱
    - Singleton 패턴으로 구현되어 리소스 효율성 보장
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
        
        # 1. 임베딩 모델 초기화 (Lazy Loading)
        # 초기 기동 속도를 위해 모델 로딩은 실제 필요 시점까지 지연
        self.model = None 

        # 2. ChromaDB 클라이언트 초기화
        if not os.path.exists(self.chroma_dir):
            os.makedirs(self.chroma_dir)
            logger.info(f"📂 ChromaDB 디렉토리 생성: {self.chroma_dir}")
            
        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        
        # 컬렉션 가져오기 또는 생성 (Cosine Similarity 사용)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"} 
        )
        
        self.kb_dir = os.path.join(settings.KNOWLEDGE_BASE_DIR, "peg_docs")
        self._initialized = True
        logger.info(f"✅ RAGService 초기화 완료 (Collection: {self.collection_name})")

    def _load_model(self):
        """임베딩 모델 로드 (Lazy Loading)"""
        if self.model is None:
            logger.info(f"⏳ 임베딩 모델 로딩 시작: {self.embedding_model_name}...")
            self.model = SentenceTransformer(self.embedding_model_name)
            logger.info("✅ 임베딩 모델 로딩 완료")

    def initialize_knowledge_base(self, force_reload: bool = False):
        """
        지식 베이스 초기화 (Initialize Knowledge Base)
        
        기술 문서(MD 파일)들을 로드하고 청킹(Chunking)하여 Vector DB에 인덱싱합니다.
        
        Args:
            force_reload (bool): 기존 데이터를 삭제하고 강제로 재구축할지 여부
        """
        try:
            # DB가 비어있지 않고 강제 리로드가 아니면 건너뜀
            if self.collection.count() > 0 and not force_reload:
                logger.info(f"ℹ️ Vector DB에 이미 {self.collection.count()}개의 항목이 있습니다. 초기화를 건너뜁니다.")
                return

            self._load_model()
            
            logger.info("📚 지식 베이스 구축 시작 (from MD files)...")
            
            # 1. 문서 로드
            md_files = glob.glob(os.path.join(self.kb_dir, "*.md"))
            
            if not md_files:
                logger.warning(f"⚠️ 지식 베이스 디렉토리에 MD 파일이 없습니다: {self.kb_dir}")
                return

            # 헤더 기반 분할 설정
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            # 문자 수 기반 추가 분할 설정
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

            ids = []
            documents = []
            metadatas = []

            for file_path in md_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    
                    # 1차: 헤더별 분할
                    header_splits = markdown_splitter.split_text(text)
                    
                    # 2차: 문자 수별 추가 분할
                    final_splits = text_splitter.split_documents(header_splits)
                    
                    for idx, split in enumerate(final_splits):
                        # 메타데이터 구성
                        source = os.path.basename(file_path)
                        meta = split.metadata.copy()
                        meta["source"] = source
                        
                        # 문서 내용
                        content = split.page_content
                        
                        # 고유 ID 생성
                        doc_id = f"{source}_{idx}"
                        
                        ids.append(doc_id)
                        documents.append(content)
                        metadatas.append(meta)
                        
                except Exception as e:
                    logger.error(f"❌ 파일 처리 중 오류 ({file_path}): {e}")
                    continue

            if not documents:
                logger.warning("⚠️ 추출된 문서 내용이 없습니다.")
                return

            # 2. 문서 임베딩
            logger.info(f"⏳ {len(documents)}개의 청크 임베딩 생성 중...")
            embeddings = self.model.encode(documents).tolist()
            
            # 3. ChromaDB 저장
            # 강제 리로드 시 기존 컬렉션 삭제 후 재생성
            if force_reload and self.collection.count() > 0:
                 logger.info("🔄 기존 컬렉션 초기화 중...")
                 self.client.delete_collection(self.collection_name)
                 self.collection = self.client.create_collection(
                    name=self.collection_name, 
                    metadata={"hnsw:space": "cosine"}
                )

            # 배치 단위로 추가 (Batch Processing)
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                end_idx = min(i + batch_size, len(ids))
                self.collection.add(
                    ids=ids[i:end_idx],
                    documents=documents[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
                
            logger.info(f"✅ 지식 베이스 인덱싱 완료: 총 {len(ids)}개 청크 저장됨.")
            
        except Exception as e:
            logger.error(f"❌ 지식 베이스 초기화 중 치명적 오류: {e}", exc_info=True)

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        문서 검색 (Search)
        
        사용자 질의(Query)와 유사도가 높은 문서를 Vector DB에서 검색합니다.
        
        Args:
            query (str): 검색 질의
            k (int): 반환할 문서 개수
            
        Returns:
            List[str]: 검색된 문서 내용 리스트
        """
        try:
            self._load_model()
            
            # 쿼리 임베딩
            query_embedding = self.model.encode([query]).tolist()
            
            # 벡터 검색
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k
            )
            
            # results['documents']는 List[List[str]] 형태임
            if results and results['documents']:
                documents = results['documents'][0]
                # logger.debug(f"🔍 검색 결과 ({k}건): {documents}") # 내용이 많아 로그 레벨 주의
                return documents
            
            return []
            
        except Exception as e:
            logger.error(f"❌ 문서 검색 중 오류: {e}", exc_info=True)
            return []

# Singleton Instance
rag_service = RAGService()
