
import re
import csv
import time
import argparse
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 1. DOCUMENTS
# The first three documents are copied from Task 2.

Docs = {
    "it_policy": (
        "The university account service helps students access email, course materials, and campus applications. "
        "Students should keep their passwords private and should never share recovery codes with another person. "
        "Rule: A forgotten password must be reset through the official account recovery link after the student verifies "
        "their registered email address. Exception: If the registered email address is no longer accessible, the student "
        "must contact the IT help desk and complete identity verification before an account recovery request can be processed. "
        "Students should use a strong, unique password and update it if they believe another person has learned it. "
        "The IT help desk can assist with locked accounts during published service hours. Students should include their "
        "university username when requesting support, but they should never send their password in an email or support ticket. "
        "A password reset link may expire, so a student who receives an expired link should begin a new recovery request "
        "from the official sign-in page. Staff will not ask students to disclose their current password. Account access may "
        "be temporarily restricted after repeated unsuccessful sign-in attempts, and the student should follow the recovery "
        "instructions rather than repeatedly guessing passwords. Students who receive an unexpected recovery notification "
        "should contact the help desk through the official university website. The university recommends checking that the "
        "browser address belongs to the official institution before entering account details. These precautions help protect "
        "student records, coursework, and personal communications from unauthorized access."
    ),

    "assessment_policy": (
        "The learning portal is the official place for submitting coursework and checking assignment status. "
        "Students should read the assessment instructions before preparing a final file. "
        "Rule: Every assignment must be uploaded to the correct course page before the deadline shown in the learning portal. "
        "Exception: If the portal is unavailable near the deadline, the student should take a screenshot of the error and email "
        "the course instructor before the deadline, then follow the instructor's submission instructions. "
        "The normal accepted format is PDF unless the assignment page specifies another file type. "
        "Students should use a clear filename containing their course code and student number. After uploading, they should "
        "wait for the confirmation message and verify that the correct file appears in the submission area. Replacing a file "
        "is permitted only while the submission link remains open. A draft saved on a personal computer is not considered a "
        "submitted assignment. If a student uploads the wrong file, they should contact the instructor promptly and explain "
        "the problem. Group assignments must identify the members according to the course instructions. Students should keep "
        "a copy of the final submitted file and the portal confirmation for their records. Late work is handled according "
        "to the course assessment rules, not by an automatic extension. The instructor may request additional evidence if "
        "a technical issue is reported. Students should not assume that an email attachment has replaced the required portal "
        "submission unless the instructor explicitly confirms that arrangement."
    ),

    "attendance_policy": (
        "Students are expected to attend scheduled classes and follow the attendance procedure published by their department. "
        "Attendance records are maintained by the course instructor or the approved university system. "
        "Rule: A student who misses a class because of illness should notify the instructor as soon as reasonably possible "
        "and provide the required supporting document when requested. Exception: A student who cannot obtain medical "
        "documentation immediately should still contact the instructor promptly and explain when the document is expected; "
        "the instructor will review the circumstances under the department's policy. Students should check the course page "
        "for notices about cancelled classes, room changes, and replacement sessions. A message sent to another student "
        "does not count as official notice to the instructor. If a student arrives late, the instructor may record the "
        "attendance according to the published class rules. Students who need an accommodation should contact the relevant "
        "university office and follow its formal process. The department may distinguish an excused absence from an absence "
        "that affects participation marks. Students should not assume that every illness automatically removes an absence "
        "from the attendance record. For a serious or extended absence, the student should communicate with the instructor "
        "and academic office so that the correct procedure can be explained. The following sentence is intentionally long "
        "for the chunking exercise: when a student is unable to attend several scheduled sessions because of a continuing "
        "personal or medical circumstance, the student should contact the department, explain the expected duration, ask "
        "which forms or supporting records are required, and keep copies of all messages so that the instructor and academic "
        "office can review the same information. Attendance decisions are made under the published course and department rules."
    ),

    "contractor_policy": (
        "Workplace remote work regulations govern remote activity for university employees and external vendors. Standard "
        "full-time university employees may request remote work arrangements up to two days per week with prior written manager approval. "
        "Rule: University staff may perform remote duties upon completing the mandatory cybersecurity compliance training. "
        "Exception: External third-party contractors and temporary staff can work remotely only if the signed contract explicitly permits it. "
        "General employee remote work rules do not apply to contractors under any circumstances. Unauthorized remote work by contractors "
        "constitutes a breach of service agreement. Managers cannot grant oral permission to contractors for remote work without an official "
        "contract amendment executed by procurement. Access to sensitive internal databases from off-campus locations requires an active encrypted "
        "VPN connection for all user categories."
    ),

    "equipment_policy": (
        "Hardware diagnostic procedures for departmental printing workstations and lab hardware. "
        "Rule: For blank printer pages, check ink levels and run the printer cleaning utility through the device control software. "
        "Print a test page afterwards to confirm nozzle alignment and output clarity. "
        "Exception: If output pages remain blank after running two consecutive cleaning cycles, replace the primary black printhead module. "
        "For paper jam errors, power off the unit before removing jammed paper from the rear feed tray to prevent gear damage. "
        "Do not use sharp metallic objects to pull jammed paper out. For network connection loss, restart the local print server "
        "service or re-bind the printer IP address."
    ),

    "remote_work_memo": (
        "A departmental operations memo describes a temporary local practice for vendor scheduling. "
        "The memo states that a department manager may approve a contractor's remote work by email for a short assignment. "
        "It says an email approval is sufficient for the department's scheduling record and does not mention a procurement "
        "contract amendment. This memo conflicts with the university contractor policy, which says a signed contract must "
        "explicitly permit contractor remote work and that a manager cannot grant oral permission without an official "
        "contract amendment executed by procurement. Staff should refer this disagreement to procurement or the policy owner."
    )
}

# 2. MANUALLY LABELLED QUERIES
# Four answerable dev queries, four answerable test queries,
# one unanswerable dev query and one unanswerable test query.

Queries = [
    {
        "query": "How can I regain access to a forgotten account password?",
        "doc": "it_policy",
        "evidence": "A forgotten password must be reset through the official account recovery link",
        "split": "dev"
    },
    {
        "query": "Where should students upload their course assignments?",
        "doc": "assessment_policy",
        "evidence": "Every assignment must be uploaded to the correct course page before the deadline",
        "split": "dev"
    },
    {
        "query": "What proof is required if a student misses class due to illness?",
        "doc": "attendance_policy",
        "evidence": "provide the required supporting document when requested",
        "split": "dev"
    },
    {
        "query": "Are external contractors allowed to work remotely?",
        "doc": "contractor_policy",
        "evidence": "External third-party contractors and temporary staff can work remotely only if the signed contract explicitly permits it",
        "split": "dev"
    },
    {
        "query": "What are the rules and prices for campus cafeteria meal plans?",
        "doc": None,
        "evidence": None,
        "split": "dev"
    },
    {
        "query": "What steps should be taken if a printer outputs blank pages?",
        "doc": "equipment_policy",
        "evidence": "For blank printer pages, check ink levels and run the printer cleaning utility",
        "split": "test"
    },
    {
        "query": "What should I do if the assignment submission portal is down near the deadline?",
        "doc": "assessment_policy",
        "evidence": "If the portal is unavailable near the deadline, the student should take a screenshot of the error",
        "split": "test"
    },
    {
        "query": "What happens if a student cannot get a medical note right away?",
        "doc": "attendance_policy",
        "evidence": "cannot obtain medical documentation immediately should still contact the instructor promptly",
        "split": "test"
    },
    {
        "query": "Can a department manager verbally authorize remote work for a contractor?",
        "doc": "contractor_policy",
        "evidence": "Managers cannot grant oral permission to contractors for remote work without an official contract amendment",
        "split": "test"
    },
    {
        "query": "What is the weekend schedule and route for the university shuttle bus?",
        "doc": None,
        "evidence": None,
        "split": "test"
    }
]

# 3. CHUNKING

def WordSpans(Text):
    return [(Match.start(), Match.end()) for Match in re.finditer(r"\S+", Text)]


def WordWindows(Text, Size=40, Overlap=0, Offset=0):
    if Size <= 0 or Overlap < 0 or Overlap >= Size:
        raise ValueError("Size must be positive and overlap must be smaller than size.")

    Spans = WordSpans(Text)
    Chunks = []

    for Start in range(0, len(Spans), Size - Overlap):
        EndIndex = min(Start + Size, len(Spans))
        Chunks.append((
            Offset + Spans[Start][0],
            Offset + Spans[EndIndex - 1][1]
        ))

        if EndIndex == len(Spans):
            break

    return Chunks


def SentenceChunks(Text, Size=40):
    if Size <= 0:
        raise ValueError("Size must be positive.")

    Cuts = [0] + [Match.end() for Match in re.finditer(r"[.!?]\s+", Text)] + [len(Text)]
    Result = []
    Pending = None
    PendingWords = 0

    for Start, End in zip(Cuts, Cuts[1:]):
        Sentence = Text[Start:End]
        WordCount = len(WordSpans(Sentence))

        if WordCount == 0:
            continue

        if WordCount > Size:
            if Pending is not None:
                Result.append(Pending)

            Result.extend(WordWindows(Sentence, Size, 0, Start))
            Pending = None
            PendingWords = 0

        elif Pending is not None and PendingWords + WordCount <= Size:
            Pending = (Pending[0], End)
            PendingWords += WordCount

        else:
            if Pending is not None:
                Result.append(Pending)

            Pending = (Start, End)
            PendingWords = WordCount

    if Pending is not None:
        Result.append(Pending)

    return Result


def MakeChunks(DocumentMap, Strategy):
    Chunks = []

    for DocId, Text in DocumentMap.items():
        if Strategy == "sentence":
            Spans = SentenceChunks(Text, 40)
        elif Strategy == "0":
            Spans = WordWindows(Text, 40, 0)
        else:
            Spans = WordWindows(Text, 40, 10)

        for Index, (Start, End) in enumerate(Spans):
            Chunks.append({
                "id": f"{DocId}:{Index}",
                "doc": DocId,
                "start": Start,
                "end": End,
                "text": Text[Start:End]
            })

    return Chunks

# 4. METRICS AND THRESHOLD

def Evaluate(Scores, Chunks, Items):
    HitValues = []
    ReciprocalValues = []

    for Row, Query in zip(Scores, Items):
        if Query["doc"] is None:
            continue

        TopIndices = np.argsort(-Row, kind="stable")[:3]
        GoodRanks = []

        for Rank, ChunkIndex in enumerate(TopIndices, 1):
            Chunk = Chunks[ChunkIndex]

            if (
                Chunk["doc"] == Query["doc"]
                and Query["evidence"] in Chunk["text"]
            ):
                GoodRanks.append(Rank)

        HitValues.append(bool(GoodRanks))
        ReciprocalValues.append(1 / GoodRanks[0] if GoodRanks else 0)

    return float(np.mean(HitValues)), float(np.mean(ReciprocalValues))


def CalibrateThreshold(Scores, Items):
    # Uses development queries only. It is not a universal cutoff.
    TopScores = np.max(Scores, axis=1)
    Labels = np.array([1 if Item["doc"] is not None else 0 for Item in Items])

    SortedScores = np.sort(TopScores)
    MidPoints = (SortedScores[:-1] + SortedScores[1:]) / 2
    Candidates = np.unique(np.concatenate((
        [0.0],
        SortedScores,
        MidPoints,
        [float(np.max(TopScores)) + 0.000001]
    )))

    BestThreshold = float(Candidates[0])
    BestF1 = -1.0

    for Threshold in Candidates:
        Predictions = (TopScores >= Threshold).astype(int)

        TruePositive = np.sum((Predictions == 1) & (Labels == 1))
        FalsePositive = np.sum((Predictions == 1) & (Labels == 0))
        FalseNegative = np.sum((Predictions == 0) & (Labels == 1))

        Precision = TruePositive / (TruePositive + FalsePositive) if TruePositive + FalsePositive else 0
        Recall = TruePositive / (TruePositive + FalseNegative) if TruePositive + FalseNegative else 0

        F1 = (
            2 * Precision * Recall / (Precision + Recall)
            if Precision + Recall else 0
        )

        if F1 > BestF1:
            BestF1 = F1
            BestThreshold = float(Threshold)

    return BestThreshold, float(BestF1)


# 5. FAQ BASELINE

FaqItems = [
    ("How do I reset my university password?",
     "Use the official recovery link. Contact the IT help desk if your registered email is inaccessible."),
    ("Where do I submit an assignment?",
     "Upload it to the correct course page in the learning portal before the deadline."),
    ("What should I do after missing class due to illness?",
     "Notify the instructor and provide supporting documentation when requested."),
    ("How can a contractor work remotely?",
     "The signed contract must explicitly permit contractor remote work."),
    ("How do I fix blank printer pages?",
     "Check ink levels and run the printer cleaning utility.")
]

# 6. MAIN EVALUATION

def Main():
    print("Documents:", len(Docs))
    print("Queries:", len(Queries))
    print("Answerable:", sum(Query["doc"] is not None for Query in Queries))
    print("Unanswerable:", sum(Query["doc"] is None for Query in Queries))

    DevQueries = [Query for Query in Queries if Query["split"] == "dev"]
    TestQueries = [Query for Query in Queries if Query["split"] == "test"]
    QueryTexts = [Query["query"] for Query in Queries]

    assert len([Q for Q in DevQueries if Q["doc"] is not None]) == 4
    assert len([Q for Q in TestQueries if Q["doc"] is not None]) == 4
    assert len(DevQueries) == 5 and len(TestQueries) == 5

    Strategies = ["0", "10", "sentence"]
    Methods = ["tfidf", "embedding"]
    ChunkSets = {Strategy: MakeChunks(Docs, Strategy) for Strategy in Strategies}

    for ChunkList in ChunkSets.values():
        for Chunk in ChunkList:
            OriginalText = Docs[Chunk["doc"]][Chunk["start"]:Chunk["end"]]
            assert OriginalText == Chunk["text"]

    LoadStart = time.perf_counter()
    Model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    ModelLoadTime = time.perf_counter() - LoadStart

    print("\nModel load time:", round(ModelLoadTime, 4), "seconds")
    print("Model token limit:", Model.max_seq_length)

    print("\nTOKEN LENGTH AUDIT")
    for Strategy, ChunkList in ChunkSets.items():
        Texts = [Chunk["text"] for Chunk in ChunkList] + QueryTexts
        TokenLengths = [
            len(Model.tokenizer(Text, truncation=False)["input_ids"])
            for Text in Texts
        ]
        MaxTokens = max(TokenLengths)
        print(
            Strategy,
            "| Maximum tokens:", MaxTokens,
            "| Model limit:", Model.max_seq_length,
            "| Fits:", MaxTokens <= Model.max_seq_length
        )

    DevIndices = [Index for Index, Q in enumerate(Queries) if Q["split"] == "dev"]
    TestIndices = [Index for Index, Q in enumerate(Queries) if Q["split"] == "test"]

    Results = []
    RunData = {}
    CsvRows = []

    for Strategy in Strategies:
        Chunks = ChunkSets[Strategy]
        ChunkTexts = [Chunk["text"] for Chunk in Chunks]

        for Method in Methods:
            EncodeStart = time.perf_counter()

            if Method == "tfidf":
                Vectorizer = TfidfVectorizer()
                ChunkVectors = Vectorizer.fit_transform(ChunkTexts)
                QueryVectors = Vectorizer.transform(QueryTexts)
            else:
                ChunkVectors = Model.encode(
                    ChunkTexts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                QueryVectors = Model.encode(
                    QueryTexts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                Vectorizer = None

            EncodeTime = time.perf_counter() - EncodeStart

            SearchStart = time.perf_counter()

            if Method == "tfidf":
                ScoreMatrix = cosine_similarity(QueryVectors, ChunkVectors)
            else:
                ScoreMatrix = QueryVectors @ ChunkVectors.T

            SearchTime = time.perf_counter() - SearchStart

            DevScores = ScoreMatrix[DevIndices]
            TestScores = ScoreMatrix[TestIndices]

            DevHit, DevMrr = Evaluate(
                DevScores, Chunks, DevQueries
            )
            TestHit, TestMrr = Evaluate(
                TestScores, Chunks, TestQueries
            )

            Threshold, DevF1 = CalibrateThreshold(DevScores, DevQueries)

            Results.append({
                "Strategy": Strategy,
                "Method": Method,
                "Dev Hit@3": DevHit,
                "Dev MRR@3": DevMrr,
                "Test Hit@3": TestHit,
                "Test MRR@3": TestMrr,
                "Threshold": Threshold,
                "Dev threshold F1": DevF1,
                "Chunk count": len(Chunks),
                "Encoding time": EncodeTime,
                "Search time": SearchTime
            })

            RunData[(Strategy, Method)] = {
                "chunks": Chunks,
                "chunk_vectors": ChunkVectors,
                "vectorizer": Vectorizer,
                "threshold": Threshold,
                "scores": ScoreMatrix
            }

            for QueryIndex, Query in enumerate(Queries):
                TopIndices = np.argsort(
                    -ScoreMatrix[QueryIndex], kind="stable"
                )[:3]

                for Rank, ChunkIndex in enumerate(TopIndices, 1):
                    Chunk = Chunks[ChunkIndex]

                    CsvRows.append({
                        "strategy": Strategy,
                        "method": Method,
                        "query": Query["query"],
                        "split": Query["split"],
                        "answerable": Query["doc"] is not None,
                        "expected_doc": Query["doc"],
                        "rank": Rank,
                        "score": float(ScoreMatrix[QueryIndex, ChunkIndex]),
                        "chunk_id": Chunk["id"],
                        "source_doc": Chunk["doc"],
                        "start_offset": Chunk["start"],
                        "end_offset": Chunk["end"],
                        "text": Chunk["text"]
                    })

    with open("task3_retrieval_log.csv", "w", newline="", encoding="utf-8") as File:
        Writer = csv.DictWriter(File, fieldnames=CsvRows[0].keys())
        Writer.writeheader()
        Writer.writerows(CsvRows)

    Results.sort(
        key=lambda Row: (
            -Row["Dev Hit@3"],
            -Row["Dev MRR@3"],
            Row["Chunk count"],
            -Row["Dev threshold F1"],
            Row["Strategy"],
            Row["Method"]
        )
    )

    print("\nRESULTS")
    for Row in Results:
        print(
            Row["Strategy"], Row["Method"],
            "| Dev Hit@3:", round(Row["Dev Hit@3"], 3),
            "| Dev MRR@3:", round(Row["Dev MRR@3"], 3),
            "| Test Hit@3:", round(Row["Test Hit@3"], 3),
            "| Test MRR@3:", round(Row["Test MRR@3"], 3),
            "| Chunks:", Row["Chunk count"],
            "| Encoding:", round(Row["Encoding time"], 4),
            "| Search:", round(Row["Search time"], 5),
            "| Threshold:", round(Row["Threshold"], 4)
        )

    Best = Results[0]
    SelectedStrategy = Best["Strategy"]
    SelectedMethod = Best["Method"]
    SelectedRun = RunData[(SelectedStrategy, SelectedMethod)]

    print("\nSELECTED USING DEVELOPMENT RESULTS ONLY")
    print("Strategy:", SelectedStrategy)
    print("Method:", SelectedMethod)
    print("Development threshold:", round(SelectedRun["threshold"], 4))

    print("\nFINAL HELD-OUT TEST RESULTS")
    print("Hit@3:", round(Best["Test Hit@3"], 4))
    print("MRR@3:", round(Best["Test MRR@3"], 4))
    print("The test results were not used to select the configuration.")

    # Print top three text, scores, IDs and source offsets.
    print("\nTOP THREE RESULTS FOR EACH QUERY")
    for QueryIndex, Query in enumerate(Queries):
        print("\nQUERY:", Query["query"], "|", Query["split"])
        RowScores = SelectedRun["scores"][QueryIndex]
        TopIndices = np.argsort(-RowScores, kind="stable")[:3]

        for Rank, ChunkIndex in enumerate(TopIndices, 1):
            Chunk = SelectedRun["chunks"][ChunkIndex]
            print(
                "\nRank:", Rank,
                "| Score:", round(float(RowScores[ChunkIndex]), 4),
                "| ID:", Chunk["id"],
                "| Offsets:", Chunk["start"], Chunk["end"]
            )
            print("Text:", Chunk["text"])

    print("\nHELD-OUT TEST THRESHOLD DECISIONS (NO RETUNING)")
    TestAccepted = 0
    TestRejected = 0
    TestFalseAccept = 0
    TestFalseReject = 0
    TestUnanswerableCount = 0
    for QueryIndex, Query in enumerate(Queries):
        if Query["split"] != "test":
            continue
        TopScore = float(np.max(SelectedRun["scores"][QueryIndex]))
        Accepted = TopScore >= SelectedRun["threshold"]
        if Accepted:
            TestAccepted += 1
        else:
            TestRejected += 1
        if Query["doc"] is None:
            TestUnanswerableCount += 1
            if Accepted:
                TestFalseAccept += 1
        elif not Accepted:
            TestFalseReject += 1
        print(
            Query["split"], "|", Query["query"],
            "| answerable:", Query["doc"] is not None,
            "| top score:", round(TopScore, 4),
            "| threshold:", round(SelectedRun["threshold"], 4),
            "| decision:", "ACCEPT" if Accepted else "REJECT",
            "| correct:", Accepted == (Query["doc"] is not None)
        )
    print("Test accepted:", TestAccepted, "| Test rejected:", TestRejected)
    print("Test false accepts:", TestFalseAccept)
    print("Test false rejects:", TestFalseReject)
    print(
        "Test false-acceptance rate:",
        round(TestFalseAccept / TestUnanswerableCount, 4)
        if TestUnanswerableCount else "not available"
    )
    print("The threshold was calibrated on development queries and was not changed for test.")

    # Explain the nearest-neighbour behavior for dev and test separately.
    print("\nWHY UNANSWERABLE QUERIES STILL RETURN TEXT")
    for SplitName in ["dev", "test"]:
        for QueryIndex, Query in enumerate(Queries):
            if Query["split"] == SplitName and Query["doc"] is None:
                TopScore = float(np.max(SelectedRun["scores"][QueryIndex]))
                Decision = "ACCEPT" if TopScore >= SelectedRun["threshold"] else "REJECT"
                BestIndex = int(np.argmax(SelectedRun["scores"][QueryIndex]))
                BestChunk = SelectedRun["chunks"][BestIndex]
                print(
                    SplitName, "unanswerable query:", Query["query"],
                    "| top candidate:", BestChunk["id"],
                    "| score:", round(TopScore, 4),
                    "| threshold decision:", Decision
                )
                if TopScore >= SelectedRun["threshold"]:
                    print("Observed failure: this unanswerable query passes the threshold (false acceptance).")
                else:
                    print("This unanswerable query is rejected by the calibrated threshold.")
                print(
                    "Reason: retrieval always ranks the top-k nearest chunks. "
                    "Cosine similarity measures relative closeness, not whether "
                    "the corpus actually contains an answer."
                )

    RulePairs = {
        "it_policy": (
            "A forgotten password must be reset",
            "If the registered email address is no longer accessible"
        ),
        "assessment_policy": (
            "Every assignment must be uploaded",
            "If the portal is unavailable near the deadline"
        ),
        "attendance_policy": (
            "misses a class because of illness",
            "cannot obtain medical documentation immediately"
        )
    }

    print("\nOBSERVED RULE AND EXCEPTION CHUNK CHECK")
    SplitExamples = []
    for Strategy, ChunkList in ChunkSets.items():
        for DocId, (RuleText, ExceptionText) in RulePairs.items():
            RuleChunks = [
                Chunk["id"] for Chunk in ChunkList
                if Chunk["doc"] == DocId and RuleText in Chunk["text"]
            ]
            ExceptionChunks = [
                Chunk["id"] for Chunk in ChunkList
                if Chunk["doc"] == DocId and ExceptionText in Chunk["text"]
            ]
            Together = any(
                RuleId == ExceptionId
                for RuleId in RuleChunks for ExceptionId in ExceptionChunks
            )
            print(
                Strategy, "|", DocId, "| together:", Together,
                "| rule chunks:", RuleChunks,
                "| exception chunks:", ExceptionChunks
            )
            if not Together and RuleChunks and ExceptionChunks:
                SplitExamples.append((Strategy, DocId, RuleChunks, ExceptionChunks))
    if SplitExamples:
        print("Observed split example:", SplitExamples[0])
    else:
        print("No rule/exception split was observed in these chunk configurations.")

    print("\nOBSERVED MANAGER/CONTRACTOR RETRIEVAL")
    ManagerQueryIndex = next(
        i for i, q in enumerate(Queries)
        if "manager" in q["query"].lower() and "contractor" in q["query"].lower()
    )
    ManagerQuery = Queries[ManagerQueryIndex]
    ManagerScores = SelectedRun["scores"][ManagerQueryIndex]
    ManagerTop = np.argsort(-ManagerScores, kind="stable")[:3]
    ManagerSources = set()
    for Rank, ChunkIndex in enumerate(ManagerTop, 1):
        Chunk = SelectedRun["chunks"][ChunkIndex]
        ManagerSources.add(Chunk["doc"])
        print(
            "Rank:", Rank, "|", Chunk["id"],
            "| score:", round(float(ManagerScores[ChunkIndex]), 4),
            "| source:", Chunk["doc"]
        )
    if "remote_work_memo" in ManagerSources and "contractor_policy" in ManagerSources:
        print("Observed conflict: the retrieved top three include both the policy and the memo.")
    elif "remote_work_memo" in ManagerSources:
        print("Observed issue: the memo appears among the top three; compare it with the contractor policy.")
    else:
        print("The memo was not in this query's top three for the selected configuration.")

    print("\nOBSERVED EVIDENCE SPLIT ACROSS CHUNKS")
    EvidenceSplitFound = False
    for QueryIndex, Query in enumerate(Queries):
        if Query["doc"] is None:
            continue
        FullDocument = Docs[Query["doc"]]
        if Query["evidence"] not in FullDocument:
            continue
        MatchingChunks = [
            Chunk["id"] for Chunk in SelectedRun["chunks"]
            if Chunk["doc"] == Query["doc"] and Query["evidence"] in Chunk["text"]
        ]
        if not MatchingChunks:
            EvidenceSplitFound = True
            RowScores = SelectedRun["scores"][QueryIndex]
            TopIndices = np.argsort(-RowScores, kind="stable")[:3]
            print("Query:", Query["query"])
            print("Expected evidence exists in the full document but crosses chunk boundaries.")
            print("Top three retrieved chunks:")
            for Rank, ChunkIndex in enumerate(TopIndices, 1):
                Chunk = SelectedRun["chunks"][ChunkIndex]
                print(
                    Rank, Chunk["id"], Chunk["doc"],
                    "score:", round(float(RowScores[ChunkIndex]), 4)
                )
            break
    if not EvidenceSplitFound:
        print("No labelled evidence phrase was split across chunks in the selected configuration.")

    print(
        "Metric note: evidence-substring scoring is strict. A relevant passage can be "
        "retrieved but counted as a miss when its full labelled evidence phrase is "
        "split across a 40-word chunk boundary."
    )

    print("\nFAILURE ANALYSIS SUMMARY")
    if SplitExamples:
        print("- Rule and exception separation was observed; see the chunk IDs above.")
    print("- The manager/contractor query was inspected against the actual top-three sources above.")
    print("- Both unanswerable examples were checked against the dev-calibrated threshold above.")
    print("- Evidence-substring misses caused by chunk boundaries are reported above when present.")
    print("- Only four answerable development queries are available, so the selected configuration and threshold are provisional.")

    print("\nFAQ MATCHING VS DOCUMENT RETRIEVAL")
    FaqVectorizer = TfidfVectorizer()
    FaqVectors = FaqVectorizer.fit_transform([Item[0] for Item in FaqItems])

    ExampleQuestions = [
        "What if my university email cannot be accessed during password recovery?",
        "Can a manager approve a contractor by email?",
        "weekend shuttle route timetable"
    ]

    for Question in ExampleQuestions:
        FaqQuery = FaqVectorizer.transform([Question])
        FaqScores = cosine_similarity(FaqQuery, FaqVectors)[0]
        FaqIndex = int(np.argmax(FaqScores))
        FaqBestScore = float(FaqScores[FaqIndex])

        if SelectedMethod == "tfidf":
            DocQuery = SelectedRun["vectorizer"].transform([Question])
            DocScores = cosine_similarity(DocQuery, SelectedRun["chunk_vectors"])[0]
        else:
            DocQuery = Model.encode(
                [Question],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )[0]
            DocScores = SelectedRun["chunk_vectors"] @ DocQuery

        BestIndex = int(np.argmax(DocScores))
        BestChunk = SelectedRun["chunks"][BestIndex]
        DocBestScore = float(DocScores[BestIndex])
        DocAccepted = DocBestScore >= SelectedRun["threshold"]

        print("\nQuestion:", Question)
        print("FAQ match:", FaqItems[FaqIndex][0])
        print("FAQ score:", round(FaqBestScore, 4))
        print("FAQ answer:", FaqItems[FaqIndex][1])
        if np.allclose(FaqScores, 0):
            print("FAQ failure: every score is zero, but argmax still selects item 0 (a password answer).")
        else:
            print("FAQ limitation: it always selects a closest predefined question and has no built-in abstention.")
        print("Document candidate:", BestChunk["doc"], BestChunk["id"])
        print("Document score:", round(DocBestScore, 4))
        print("Document threshold:", round(SelectedRun["threshold"], 4))
        print("Document decision:", "ACCEPT for evidence review" if DocAccepted else "REJECT as insufficient evidence")
        if "manager" in Question.lower():
            print("Context/conflict note: a short FAQ answer may omit the memo and policy disagreement; document retrieval can expose both sources for review.")
        if "shuttle" in Question.lower():
            print("Missing-context note: the corpus contains no shuttle timetable, so a returned nearest passage is not evidence of a shuttle answer.")

    return {
        "model": Model,
        "strategy": SelectedStrategy,
        "method": SelectedMethod,
        "run": SelectedRun
    }


if __name__ == "__main__":
    Parser = argparse.ArgumentParser(description="Evaluate the document retrieval system.")
    Parser.add_argument(
        "--chat",
        action="store_true",
        help="Start the interactive chatbot after evaluation."
    )
    Arguments = Parser.parse_args()

    SelectedConfiguration = Main()

    if Arguments.chat:
        from document_chatbot import configure, document_chatbot
        configure(SelectedConfiguration)
        document_chatbot()
