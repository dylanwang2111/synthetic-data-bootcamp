"""Interactive RAG chatbot demo over the synthetic 10-K corpus (ad hoc, built on
top of Phase 9's retrievers -- see src.retrieval and src.step12_evaluate_retrievers).

Retrieval: BM25 + the real semantic embedding retriever (fastembed /
minishlab/potion-retrieval-32M), fused via reciprocal rank fusion -- the exact
hybrid_rrf pipeline step12 evaluates, over the 20 synthetic companies
(data/synthetic), chunked the same way src.chunks chunks them for step11/step12, so
a retrieved chunk_id lines up with data/qa_pairs.csv's gold_chunk_id.

Generation: if ANTHROPIC_API_KEY is set, asks Claude to answer using only the
retrieved passages, citing which one(s) it drew from. If not set, the chatbot still
works -- it just shows the top retrieved passage itself instead of a synthesized
answer, same as step08_generate_filings' "retrieval doesn't need a key; generation
does" fallback.

Two modes:
    python -m src.chatbot            interactive chat loop
    python -m src.chatbot --demo     scripted scenario-analysis walkthrough: one
                                      analytical question per non-baseline scenario
                                      company (13 of data/qa_pairs.csv's 15 analytical
                                      rows span 13 companies), showing whether hybrid
                                      retrieval actually lands on the scenario-
                                      grounded passage, alongside each company's own
                                      scenario summary from its spec.json.
"""

import argparse
import csv
import json
import os

import anthropic

from src.chunks import Chunk, load_chunks
from src.config import RAW_DIR
from src.retrieval import Bm25Retriever, SemanticRetriever, reciprocal_rank_fusion
from src.utils import setup_logging


logger = setup_logging("chatbot")

DATA_DIR = RAW_DIR.parent
SYNTHETIC_DIR = DATA_DIR / "synthetic"
QA_PATH = DATA_DIR / "qa_pairs.csv"

ANSWER_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 1024
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a RAG assistant answering questions about a synthetic SEC 10-K corpus "
    "of fictional companies. Answer ONLY using the numbered passages given in the "
    "user message -- if they don't contain the answer, say so plainly rather than "
    "guessing or using outside knowledge. Cite which passage(s) you used by their "
    "[n] label. Be concise."
)


def build_index() -> tuple[dict[str, Chunk], Bm25Retriever, SemanticRetriever]:
    """Load the 20 synthetic companies' chunks and build BM25 + semantic retrievers."""
    chunks = load_chunks(sources={"synthetic"})
    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    logger.info("Indexing %d chunks from %d synthetic companies", len(chunks), len({c.ticker for c in chunks}))
    bm25 = Bm25Retriever(chunk_ids, texts)
    semantic = SemanticRetriever(chunk_ids, texts)
    return {c.chunk_id: c for c in chunks}, bm25, semantic


def retrieve(
    question: str,
    chunks_by_id: dict[str, Chunk],
    bm25: Bm25Retriever,
    semantic: SemanticRetriever,
    k: int = TOP_K,
) -> list[Chunk]:
    """Rank with BM25 and semantic separately, fuse via RRF, return the top-k chunks."""
    fused = reciprocal_rank_fusion([bm25.rank(question), semantic.rank(question)])
    return [chunks_by_id[cid] for cid in fused[:k]]


def format_context(passages: list[Chunk]) -> str:
    blocks = [
        f"[{i}] {c.company_name} ({c.ticker}), FY{c.fiscal_year}, Item {c.item}:\n{c.text}"
        for i, c in enumerate(passages, start=1)
    ]
    return "\n\n".join(blocks)


def synthesize_answer(client: "anthropic.Anthropic", question: str, passages: list[Chunk]) -> str:
    prompt = f"Question: {question}\n\nPassages:\n{format_context(passages)}"
    response = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def get_client() -> "anthropic.Anthropic | None":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=api_key) if api_key else None


def answer_question(
    question: str,
    chunks_by_id: dict[str, Chunk],
    bm25: Bm25Retriever,
    semantic: SemanticRetriever,
    client: "anthropic.Anthropic | None",
) -> tuple[str, list[Chunk]]:
    passages = retrieve(question, chunks_by_id, bm25, semantic)
    if client is not None:
        return synthesize_answer(client, question, passages), passages
    return passages[0].text, passages


def print_sources(passages: list[Chunk]) -> None:
    print("Sources:")
    for i, c in enumerate(passages, start=1):
        print(f"  [{i}] {c.ticker} FY{c.fiscal_year} item {c.item} ({c.chunk_id})")


def load_scenario_info() -> dict[str, dict]:
    """ticker -> {"company_name", "type", "summary"}, read from each spec.json."""
    info = {}
    for spec_path in sorted(SYNTHETIC_DIR.glob("*/*/spec.json")):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        scenario = spec.get("scenario") or {}
        info[spec["fictional_id"]] = {
            "company_name": spec["fictional_name"],
            "type": scenario.get("type", "baseline"),
            "summary": scenario.get("summary", ""),
        }
    return info


def load_analytical_qa() -> list[dict]:
    with open(QA_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["style"] == "analytical" and r["source"] == "synthetic"]


def run_demo(
    chunks_by_id: dict[str, Chunk],
    bm25: Bm25Retriever,
    semantic: SemanticRetriever,
    client: "anthropic.Anthropic | None",
) -> None:
    """One analytical question per scenario company: does hybrid retrieval land on
    the scenario-grounded passage, not just a keyword-overlapping one?"""
    scenario_info = load_scenario_info()
    seen_tickers: set[str] = set()

    print("=" * 78)
    print("SCENARIO ANALYSIS DEMO")
    print("20 synthetic companies: 13 carry a deliberate scenario (M&A, margin crisis,")
    print("cash burn, regulatory pressure, ...); 7 are a baseline control group.")
    print("Each block: one company's scenario, an analytical question about it, and")
    print("whether hybrid (BM25 + semantic, RRF-fused) retrieval finds the grounded passage.")
    print("=" * 78)

    for row in load_analytical_qa():
        ticker = row["ticker"]
        if ticker in seen_tickers:
            continue  # one question per company caps the demo at 13 companies, not 15 rows
        seen_tickers.add(ticker)
        info = scenario_info.get(ticker, {})

        print(f"\n--- {info.get('company_name', ticker)} ({ticker}) -- scenario: {info.get('type', 'unknown')} ---")
        if info.get("summary"):
            print(f"Scenario: {info['summary']}")
        print(f"\nQ: {row['question']}")

        passages = retrieve(row["question"], chunks_by_id, bm25, semantic)
        hit = row["gold_chunk_id"] in {c.chunk_id for c in passages}
        print(f"Retrieval: {'HIT' if hit else 'MISS'} -- gold chunk {row['gold_chunk_id']} is "
              f"{'' if hit else 'NOT '}in the top-{len(passages)} fused ranking")

        if client is not None:
            print(f"\nA: {synthesize_answer(client, row['question'], passages)}")
        else:
            print(f"\nReference answer (data/qa_pairs.csv): {row['answer']}")
        print()
        print_sources(passages)

    print("\n" + "=" * 78)
    print(f"{len(seen_tickers)}/13 scenario companies covered.")
    if client is None:
        print("Set ANTHROPIC_API_KEY to see Claude-synthesized answers instead of the reference ones.")


def run_chat(
    chunks_by_id: dict[str, Chunk],
    bm25: Bm25Retriever,
    semantic: SemanticRetriever,
    client: "anthropic.Anthropic | None",
) -> None:
    print("SEC 10-K synthetic corpus chatbot -- ask about any of the 20 fictional companies.")
    print("Type 'quit' or 'exit' to leave.\n")
    if client is None:
        print("(ANTHROPIC_API_KEY not set -- answers show the top-retrieved passage, not a synthesized one.)\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break

        answer, passages = answer_question(question, chunks_by_id, bm25, semantic, client)
        print(f"\nBot: {answer}\n")
        print_sources(passages)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the scripted scenario-analysis walkthrough instead of an interactive chat.",
    )
    args = parser.parse_args()

    chunks_by_id, bm25, semantic = build_index()
    client = get_client()
    if client is None:
        logger.warning("ANTHROPIC_API_KEY not set -- running retrieval-only (no synthesized answers)")

    if args.demo:
        run_demo(chunks_by_id, bm25, semantic, client)
    else:
        run_chat(chunks_by_id, bm25, semantic, client)


if __name__ == "__main__":
    main()
