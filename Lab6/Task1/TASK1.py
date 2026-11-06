import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. TOKENIZATION COMPARISON

sample_texts = [
    "AI, search!",
    "I can't log in.",
    "help@example.com",
    "CS4944 version 2.1",
    "",
    "میرا نام احمد ہے۔"
]

def whitespace_tokenize(text):
    return text.split()


def regex_tokenize(text):
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


for number, text in enumerate(sample_texts, start=1):
    tokens1 = whitespace_tokenize(text)
    tokens2 = regex_tokenize(text)

    print(f"EXAMPLE {number}")
    print("Original Text:", repr(text))
    print("Whitespace Tokens:", tokens1)
    print("Whitespace Count:", len(tokens1))
    print("Regex Tokens:", tokens2)
    print("Regex Count:", len(tokens2))
    print("-" * 50)

assert regex_tokenize("") == []
print("Empty string test passed.")


# 2. BUILD FAQ CHATBOT USING TF-IDF

faqs = [
    {
        "question": "How do I reset my password?",
        "answer": "Open the sign-in page, select 'Forgot password', and follow the instructions sent to your registered email.",
        "source": "Student IT Help Guide"
    },
    {
        "question": "How can I submit my assignment?",
        "answer": "Open your course page, select the relevant assignment, upload your file, and confirm submission.",
        "source": "Learning Management System Guide"
    },
    {
        "question": "What should I do if I am ill and may submit late?",
        "answer": "Contact your course instructor as soon as possible and explain your situation. Follow the instructor's guidance about late submission.",
        "source": "Course Submission Guidance"
    },
    {
        "question": "How do I connect to campus Wi-Fi?",
        "answer": "Choose the official campus Wi-Fi network and follow the login instructions provided by your institution's IT department.",
        "source": "Campus IT Help Desk"
    },
    {
        "question": "What are the library opening hours?",
        "answer": "Please check the library's official notice or website for the current opening hours.",
        "source": "University Library Information"
    }
]

questions = [item["question"] for item in faqs]

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

faq_matrix = vectorizer.fit_transform(questions)


def answer_question(user_question, threshold=0.15, top_n=2):

    question = user_question.strip()

    if not question:
        return "Please enter a question."

    # Check for an exact match
    for item in faqs:
        if question.casefold() == item["question"].casefold():
            return (
                f"Answer: {item['answer']}\n"
                f"Source: {item['source']}"
            )

    # Find similar questions
    query_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        query_vector, faq_matrix
    ).flatten()

    best_matches = scores.argsort()[::-1][:top_n]

    if scores[best_matches[0]] < threshold:
        return (
            "No suitable answer was found. "
            "Please try asking in a different way or contact the relevant department."
        )

    suggestions = []

    for index in best_matches:
        if scores[index] >= threshold:
            suggestions.append(
                f"- {faqs[index]['question']} "
                f"(similarity: {scores[index]:.3f})"
            )

    return (
        "Possible matching questions:\n"
        + "\n".join(suggestions)
    )


print("\nFAQ chatbot is ready!")


# 3. CHATBOT EXECUTION

while True:

    user_input = input("\nYou: ")

    if user_input.strip().casefold() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = answer_question(user_input)

    print("Chatbot:", response)
