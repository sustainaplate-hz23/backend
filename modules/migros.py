import os
from modules.llm import BamLLM
from llama_index import ServiceContext
from llama_index.storage.storage_context import StorageContext, SimpleDocumentStore, SimpleIndexStore, SimpleVectorStore
from llama_index import load_index_from_storage
from llama_index.retrievers import VectorIndexRetriever


class MigrosRetriever:
    def __init__(self):
        llm = BamLLM()
        self.sc = ServiceContext.from_defaults(
            llm=llm,
            context_window=int(os.getenv('CONTEXT_WINDOW')),
            num_output=int(os.getenv('MAX_OUTPUT_TOKENS'))
        )
        self.query_engine = None
        self.index = None

    def load_index(self):
        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir='modules/migros_data_v3'),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir='modules/migros_data_v3'),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir='modules/migros_data_v3'),
        )
        index = load_index_from_storage(
            storage_context=storage_context,
            service_context=self.sc
        )
        self.index = index

    @staticmethod
    def recipe_to_str(recipe_doc):
        pass

    def query(self, ingredients):
        prompt = f"""
        Recipe which contains most of the following ingredients: {", ".join(ingredients)}
        """

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=30,
        )

        docs = retriever.retrieve(prompt)
        return [{"doc": doc.text, "score": doc.score} for doc in docs]
