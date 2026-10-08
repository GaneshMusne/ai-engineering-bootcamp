import asyncio
import os
from unittest.mock import patch

from api.agents.retrieval_generation import GEMINI_MODEL, rag_pipeline
from qdrant_client import QdrantClient
from langsmith import Client

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Ragas 0.3.1 applies nest_asyncio during import. Its patched tasks break
# Python 3.14 task detection and asyncio.timeout. This script uses native
# async APIs, so prevent the import-time patch and retain the standard loop.
with patch("nest_asyncio.apply"):
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, ResponseRelevancy


ls_client = Client()
qdrant_client = QdrantClient(url="http://localhost:6333")

ragas_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model=os.getenv("EVAL_GEMINI_MODEL", GEMINI_MODEL),
        google_api_key=os.environ["GOOGLE_API_KEY"],
        vertexai=False,
        max_retries=6,
    )
)
ragas_embeddings = LangchainEmbeddingsWrapper(
    GoogleGenerativeAIEmbeddings(
        model=os.getenv("EVAL_EMBEDDING_MODEL", "gemini-embedding-001"),
        google_api_key=os.environ["GOOGLE_API_KEY"],
        vertexai=False,
        task_type="SEMANTIC_SIMILARITY",
    )
)


def ragas_context_precision_id_based(run, example):
    retrieved_ids = set(map(str, run.outputs["retrieved_context_ids"]))
    reference_ids = set(map(str, example.outputs["reference_context_ids"]))
    return len(retrieved_ids & reference_ids) / len(retrieved_ids) if retrieved_ids else 0.0


def ragas_context_recall_id_based(run, example):
    retrieved_ids = set(map(str, run.outputs["retrieved_context_ids"]))
    reference_ids = set(map(str, example.outputs["reference_context_ids"]))
    return len(retrieved_ids & reference_ids) / len(reference_ids) if reference_ids else 0.0


async def ragas_faithfulness(run, example):

    sample = SingleTurnSample(
            user_input=run.outputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_context"]
        )

    scorer = Faithfulness(llm=ragas_llm)
    
    return await scorer.single_turn_ascore(sample)


async def ragas_relevancy(run, example):

    sample = SingleTurnSample(
            user_input=run.outputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_context"]
        )

    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    
    return await scorer.single_turn_ascore(sample)


async def evaluation_target(inputs):
    return await asyncio.to_thread(rag_pipeline, inputs["question"], qdrant_client)


async def main():
    return await ls_client.aevaluate(
        evaluation_target,
        data="rag-evaluation-dataset",
        evaluators=[
            ragas_context_precision_id_based,
            ragas_context_recall_id_based,
            ragas_faithfulness,
            ragas_relevancy,
        ],
        max_concurrency=2,
        experiment_prefix="retriever",
    )


if __name__ == "__main__":
    results = asyncio.run(main())
