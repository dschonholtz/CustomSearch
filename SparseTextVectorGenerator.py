"""
You should treat this as psuedocode.

This shows how you can go from a document to a tokenized document, to vectors for each of those documents that you can query against.
"""

# Import necessary libraries
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

# Ensure you have the necessary NLTK resources
nltk.download("punkt")


# Function to tokenize documents
def tokenize_documents(documents):
    return [word_tokenize(doc.lower()) for doc in documents]


# Function to create BM25 model
def create_bm25_model(tokenized_docs):
    return BM25Okapi(tokenized_docs)


# Function to query BM25 model
def query_bm25(bm25_model, query, top_n=None):
    tokenized_query = word_tokenize(query.lower())
    scores = bm25_model.get_scores(tokenized_query)
    if top_n:
        top_docs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_n
        ]
        return [(i + 1, scores[i]) for i in top_docs]
    return scores


# Main function to use BM25 with sample data
def main():
    # Sample documents
    text_documents = [
        "Natural language processing enables computers to understand human language.",
        "AI and machine learning are transforming many industries.",
        "Natural language processing techniques are evolving rapidly.",
    ]

    # Tokenize the documents
    tokenized_docs = tokenize_documents(text_documents)

    # Create a BM25 object
    bm25 = create_bm25_model(tokenized_docs)

    # Define a query
    query = "How does natural language processing work?"

    # Get scores for the query
    scores = query_bm25(bm25, query)

    # Print scores for each document
    for i, score in enumerate(scores):
        print(f"Document {i + 1}: Score = {score}")

    # Optionally, retrieve top-N documents with the highest scores
    top_n_results = query_bm25(bm25, query, top_n=2)
    print("\nTop N Documents:")
    for doc_index, score in top_n_results:
        print(f"Document {doc_index} with score: {score}")


if __name__ == "__main__":
    main()
