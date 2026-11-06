
# Install required libraries
!pip install scikit-learn sentence-transformers

# ==========================================
# 1. SYNTHETIC DOCUMENT CORPUS
# ==========================================

docs = {
    "it_policy": "The university account service helps students access email, course materials, and campus applications. Students should keep their passwords private and should never share recovery codes with another person. Rule: A forgotten password must be reset through the official account recovery link after the student verifies their registered email address. Exception: If the registered email address is no longer accessible, the student must contact the IT help desk and complete identity verification before an account recovery request can be processed. Students should use a strong, unique password and update it if they believe another person has learned it. The IT help desk can assist with locked accounts during published service hours. Students should include their university username when requesting support, but they should never send their password in an email or support ticket. A password reset link may expire, so a student who receives an expired link should begin a new recovery request from the official sign-in page. Staff will not ask students to disclose their current password. Account access may be temporarily restricted after repeated unsuccessful sign-in attempts, and the student should follow the recovery instructions rather than repeatedly guessing passwords. Students who receive an unexpected recovery notification should contact the help desk through the official university website. The university recommends checking that the browser address belongs to the official institution before entering account details. These precautions help protect student records, coursework, and personal communications from unauthorized access.",

    "assessment_policy": "The learning portal is the official place for submitting coursework and checking assignment status. Students should read the assessment instructions before preparing a final file. Rule: Every assignment must be uploaded to the correct course page before the deadline shown in the learning portal. Exception: If the portal is unavailable near the deadline, the student should take a screenshot of the error and email the course instructor before the deadline, then follow the instructor's submission instructions. The normal accepted format is PDF unless the assignment page specifies another file type. Students should use a clear filename containing their course code and student number. After uploading, they should wait for the confirmation message and verify that the correct file appears in the submission area. Replacing a file is permitted only while the submission link remains open. A draft saved on a personal computer is not considered a submitted assignment. If a student uploads the wrong file, they should contact the instructor promptly and explain the problem. Group assignments must identify the members according to the course instructions. Students should keep a copy of the final submitted file and the portal confirmation for their records. Late work is handled according to the course assessment rules, not by an automatic extension. The instructor may request additional evidence if a technical issue is reported. Students should not assume that an email attachment has replaced the required portal submission unless the instructor explicitly confirms that arrangement.",

    "attendance_policy": "Students are expected to attend scheduled classes and follow the attendance procedure published by their department. Attendance records are maintained by the course instructor or the approved university system. Rule: A student who misses a class because of illness should notify the instructor as soon as reasonably possible and provide the required supporting document when requested. Exception: A student who cannot obtain medical documentation immediately should still contact the instructor promptly and explain when the document is expected; the instructor will review the circumstances under the department's policy. Students should check the course page for notices about cancelled classes, room changes, and replacement sessions. A message sent to another student does not count as official notice to the instructor. If a student arrives late, the instructor may record the attendance according to the published class rules. Students who need an accommodation should contact the relevant university office and follow its formal process. The department may distinguish an excused absence from an absence that affects participation marks. Students should not assume that every illness automatically removes an absence from the attendance record. For a serious or extended absence, the student should communicate with the instructor and academic office so that the correct procedure can be explained. The following sentence is intentionally long for the chunking exercise: when a student is unable to attend several scheduled sessions because of a continuing personal or medical circumstance, the student should contact the department, explain the expected duration, ask which forms or supporting records are required, and keep copies of all messages so that the instructor and academic office can review the same information. Attendance decisions are made under the published course and department rules."
}

for doc_id, text in docs.items():
    print(doc_id, len(text.split()), "words")


# ==========================================
# 2. FIND WORD SPANS
# ==========================================

import re

def word_spans(text):
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


# ==========================================
# 3. CREATE FIXED-SIZE WORD CHUNKS
# ==========================================

def windows(text, size=40, overlap=10, offset=0):

    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")

    spans = word_spans(text)
    result = []

    for i in range(0, len(spans), size - overlap):
        j = min(i + size, len(spans))

        result.append((
            offset + spans[i][0],
            offset + spans[j - 1][1]
        ))

        if j == len(spans):
            break

    return result


# ==========================================
# 4. CREATE SENTENCE-AWARE CHUNKS
# ==========================================

def sentence_chunks(text, size=40):

    if size <= 0:
        raise ValueError("size must be positive")

    cuts = [0] + [
        match.end() for match in re.finditer(r"[.!?]\s+", text)
    ] + [len(text)]

    output = []
    pending = None
    count = 0

    for a, b in zip(cuts, cuts[1:]):

        segment = text[a:b]
        number_of_words = len(segment.split())

        if not number_of_words:
            continue

        if number_of_words > size:

            if pending is not None:
                output.append(pending)

            output.extend(windows(segment, size, 0, a))
            pending = None
            count = 0

        elif pending is not None and count + number_of_words <= size:

            pending = (pending[0], b)
            count += number_of_words

        else:

            if pending is not None:
                output.append(pending)

            pending = (a, b)
            count = number_of_words

    if pending is not None:
        output.append(pending)

    return output


# ==========================================
# 5. BUILD CHUNK RECORDS
# ==========================================

def make_chunks(docs, strategy):

    result = []

    for doc_id, text in docs.items():

        if strategy == "sentence":
            spans = sentence_chunks(text, 40)
        else:
            spans = windows(text, 40, int(strategy))

        for i, (a, b) in enumerate(spans):

            result.append({
                "id": f"{doc_id}:{i}",
                "doc": doc_id,
                "start": a,
                "end": b,
                "text": text[a:b]
            })

    return result


# ==========================================
# 6. COMPARE THREE CHUNKING STRATEGIES
# ==========================================

strategies = ["0", "10", "sentence"]

chunk_sets = {
    strategy: make_chunks(docs, strategy)
    for strategy in strategies
}

original_word_total = sum(
    len(text.split()) for text in docs.values()
)

for strategy, chunks in chunk_sets.items():

    sizes = [len(chunk["text"].split()) for chunk in chunks]

    emitted_words = sum(sizes)
    duplicated_words = emitted_words - original_word_total

    print("Strategy:", strategy)
    print("Chunk count:", len(chunks))
    print("Smallest chunk:", min(sizes))
    print("Largest chunk:", max(sizes))
    print("Average chunk size:", round(sum(sizes) / len(sizes), 2))
    print("Original word units:", original_word_total)
    print("Emitted word units:", emitted_words)
    print("Repeated word units:", duplicated_words)
    print("-" * 55)


# ==========================================
# 7. CHECK RULE AND EXCEPTION TOGETHER
# ==========================================

rule_exception_pairs = {

    "it_policy": (
        "Rule: A forgotten password must be reset through the official account recovery link after the student verifies their registered email address.",
        "Exception: If the registered email address is no longer accessible, the student must contact the IT help desk and complete identity verification before an account recovery request can be processed."
    ),

    "assessment_policy": (
        "Rule: Every assignment must be uploaded to the correct course page before the deadline shown in the learning portal.",
        "Exception: If the portal is unavailable near the deadline, the student should take a screenshot of the error and email the course instructor before the deadline, then follow the instructor's submission instructions."
    ),

    "attendance_policy": (
        "Rule: A student who misses a class because of illness should notify the instructor as soon as reasonably possible and provide the required supporting document when requested.",
        "Exception: A student who cannot obtain medical documentation immediately should still contact the instructor promptly and explain when the document is expected; the instructor will review the circumstances under the department's policy."
    )
}

for strategy, chunks in chunk_sets.items():

    print("Strategy:", strategy)

    for doc_id, (rule, exception) in rule_exception_pairs.items():

        together = any(
            chunk["doc"] == doc_id
            and rule in chunk["text"]
            and exception in chunk["text"]
            for chunk in chunks
        )

        print(doc_id, "rule and exception together:", together)

    print("-" * 55)


# ==========================================
# 8. INSPECT CHUNK IDS, OFFSETS AND TEXT
# ==========================================

for strategy in strategies:

    print("STRATEGY:", strategy)

    for chunk in chunk_sets[strategy][:5]:

        original = docs[chunk["doc"]][
            chunk["start"]:chunk["end"]
        ]

        print("ID:", chunk["id"])
        print("Source:", chunk["doc"])
        print("Offsets:", (chunk["start"], chunk["end"]))
        print("Exact source match:", original == chunk["text"])
        print("Text:", chunk["text"])
        print("-" * 50)


# ==========================================
# 9. BOUNDARY AND EDGE-CASE TESTS
# ==========================================

print("Empty text:", windows(""))
print("Short text:", windows("short text"))

try:
    windows("a b", size=2, overlap=2)

except ValueError as error:
    print("Invalid overlap:", error)


long_sentence = " ".join(["word"] * 95) + "."

long_sentence_chunks = sentence_chunks(long_sentence, 40)

print(
    "Long sentence chunk sizes:",
    [
        len(long_sentence[a:b].split())
        for a, b in long_sentence_chunks
    ]
)

for strategy in strategies:

    for chunk in chunk_sets[strategy]:

        assert chunk["text"] == docs[chunk["doc"]][
            chunk["start"]:chunk["end"]
        ]

assert windows("") == []
assert windows("short text") == [(0, 10)]

assert all(
    len(long_sentence[a:b].split()) <= 40
    for a, b in long_sentence_chunks
)

print("Boundary checks completed.")


# ==========================================
# 10. COMPARE TF-IDF AND SEMANTIC SUGGESTIONS
# ==========================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

retrieval_chunks = chunk_sets["sentence"]

chunk_texts = [
    chunk["text"] for chunk in retrieval_chunks
]

queries = [

    {
        "query": "How can I regain access to my account?",
        "expected_source": "it_policy"
    },

    {
        "query": "Where should I hand in my coursework?",
        "expected_source": "assessment_policy"
    },

    {
        "query": "What proof is needed if I was sick?",
        "expected_source": "attendance_policy"
    }
]


# TF-IDF vectors

tfidf_vectorizer = TfidfVectorizer()

tfidf_chunk_vectors = tfidf_vectorizer.fit_transform(chunk_texts)

tfidf_query_vectors = tfidf_vectorizer.transform(
    [item["query"] for item in queries]
)


# Sentence embeddings

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

embedding_chunk_vectors = model.encode(
    chunk_texts,
    normalize_embeddings=True
)

embedding_query_vectors = model.encode(
    [item["query"] for item in queries],
    normalize_embeddings=True
)


# Calculate similarity scores

tfidf_scores = cosine_similarity(
    tfidf_query_vectors,
    tfidf_chunk_vectors
)

embedding_scores = (
    embedding_query_vectors @ embedding_chunk_vectors.T
)

print("Embedding model loaded and vectors created.")


# ==========================================
# 11. DISPLAY TOP SUGGESTIONS AND EVIDENCE
# ==========================================

def show_suggestions(
    scores,
    query_items,
    chunks,
    method_name,
    top_k=3
):

    for row_index, query_item in enumerate(query_items):

        ranked_indices = np.argsort(
            -scores[row_index],
            kind="stable"
        )[:top_k]

        print("Method:", method_name)
        print("Query:", query_item["query"])
        print("Expected source:", query_item["expected_source"])

        for rank, chunk_index in enumerate(
            ranked_indices,
            start=1
        ):

            chunk = chunks[chunk_index]

            print("Rank:", rank)
            print("Chunk ID:", chunk["id"])
            print("Source document:", chunk["doc"])

            print(
                "Character offsets:",
                (chunk["start"], chunk["end"])
            )

            print(
                "Similarity score:",
                round(float(scores[row_index, chunk_index]), 4)
            )

            print(
                "Expected source match:",
                chunk["doc"] == query_item["expected_source"]
            )

            print("Original evidence:", chunk["text"])
            print("-" * 60)


# Display TF-IDF results

show_suggestions(
    tfidf_scores,
    queries,
    retrieval_chunks,
    "TF-IDF"
)


# Display sentence embedding results

show_suggestions(
    embedding_scores,
    queries,
    retrieval_chunks,
    "Sentence embeddings"
)
