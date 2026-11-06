
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

selected_config = None


# Configure the selected retrieval system
def configure(config):
    global selected_config
    selected_config = config


# Search for the most relevant document chunks
def search_selected(query_text, top_k=3):

    if selected_config is None:
        raise RuntimeError(
            "Call configure() with the selected configuration first."
        )

    if not query_text.strip():
        return []

    model = selected_config["model"]
    selected_run = selected_config["run"]
    method = selected_config["method"]

    chunks = selected_run["chunks"]
    chunk_vectors = selected_run["chunk_vectors"]
    vectorizer = selected_run["vectorizer"]

    # TF-IDF search
    if method == "tfidf":

        query_vector = vectorizer.transform([query_text])

        scores = cosine_similarity(
            query_vector,
            chunk_vectors
        )[0]

    # Semantic search
    else:

        query_vector = model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )[0]

        scores = chunk_vectors @ query_vector

    # Select top matching chunks
    top_indices = np.argsort(
        -scores,
        kind="stable"
    )[:top_k]

    return [
        (chunks[index], float(scores[index]))
        for index in top_indices
    ]


# Run the document chatbot
def document_chatbot():

    if selected_config is None:
        raise RuntimeError(
            "Call configure() with the selected configuration first."
        )

    strategy = selected_config["strategy"]
    method = selected_config["method"]
    threshold = selected_config["run"]["threshold"]

    print("\n" + "=" * 72)
    print("UNIVERSITY DOCUMENT EVIDENCE ASSISTANT")
    print("Selected:", strategy, "|", method)
    print("Type 'exit' to stop.")
    print("=" * 72)

    while True:

        try:
            question = input("\nYour question: ").strip()

        except EOFError:
            print("\nNo interactive input is available. Chatbot stopped.")
            break

        if question.lower() in ["exit", "quit"]:
            print("Chatbot stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        # Retrieve top three matching chunks
        hits = search_selected(question, top_k=3)

        if not hits:
            print("No candidate chunks were found.")
            continue

        print("\nTop candidate results:")

        for rank, (chunk, score) in enumerate(hits, 1):

            print(
                rank, "|", chunk["id"],
                "| Score:", round(score, 4),
                "| Source:", chunk["doc"],
                "| Offsets:",
                f'{chunk["start"]}:{chunk["end"]}'
            )

        best_chunk, best_score = hits[0]

        # Check the minimum similarity threshold
        if best_score < threshold:

            print("\nNot enough evidence was found. No answer will be asserted.")

            print(
                "The nearest-neighbour search still returns candidates "
                "even when the corpus has no answer."
            )

            continue

        # Check whether retrieved sources conflict
        source_names = {
            chunk["doc"] for chunk, score in hits
        }

        if (
            "contractor_policy" in source_names
            and "remote_work_memo" in source_names
        ):

            print("\nWarning: Retrieved sources conflict.")

            print(
                "The chatbot cannot determine which source has higher authority."
            )

            print(
                "Confirm the disagreement with procurement or the policy owner."
            )

        # Ask before displaying original evidence
        try:

            confirm = input(
                "\nDisplay the retrieved source evidence? (yes/no): "
            ).strip().lower()

        except EOFError:

            print(
                "\nNo confirmation input is available. "
                "Evidence was not displayed."
            )

            break

        if confirm in ["yes", "y"]:

            for rank, (chunk, score) in enumerate(hits, 1):

                print("\nEvidence", rank)
                print("Document:", chunk["doc"])
                print("Chunk ID:", chunk["id"])

                print(
                    "Character offsets:",
                    chunk["start"],
                    "to",
                    chunk["end"]
                )

                print("Score:", round(score, 4))
                print("Text:", chunk["text"])

        else:

            print("Evidence was not displayed. No answer was asserted.")


# Main program
if __name__ == "__main__":

    from task3_retrieval import Main

    configure(Main())

    document_chatbot()
