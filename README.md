# General Caveat

I did everything in here in about 2 hours.
It should be looked at as some thoughts on how to slap together the tools.
It should be assumed that any thoughts on infrastructure and scalability would have to be considered separately.

# Searching an Orgs Images and Documents

The purpose of this repository is to outline how you would go about building out a complete search system across images and text.

Some caveats, this is for defining a POC, and not something scalable yet.

## Requirements

1. Search text with image or text
2. Search images with image or text

Out of scope:

1. Filtering what you are searching. This should be easy to do later
2. Scale. This will not be easy. But will be doable.

## Files in this repo

The python here is all pseudocode and should be treated as an introduction to the tools not as something you would actually want to use.

[OCRMagic.py](OCRMagic.py) - This will help you do OCR on arbitrary images. Really naive take. You also might just want to do text extraction that has semantic understanding with an LLM. Rather than just naively using tesseract as I suggest in that code.

[Server.py](Server.py) - This is a set of FastAPI webserver api stubs. I find it easier to read the python code, but you can also see the [OpenAPI](SearchManagementOpenApi.yaml) spec if you prefer.

[SparseTextVectorGenerator.py](SparseTextVectorGenerator.py) - This introduces you to the ideas of tokenization and some other basic NLP concepts for sparse representation of text.

[TrainEmbeddingModel.py](TrainEmbeddingModel.py) - This shows you how to load the open source CLIP model by openAI to show you how to finetune an existing embedding model with your own data. This assumes you need a multi-modal, image and text, embedding model. In practice, this of course becomes a large data engineering task with your own data pipeline compared to this very simple script.

## Search Pipeline

1. Collect the relevant data you want to be able to search
2. Vectorize this data
3. Build infrastructure to add and evict searchable data
4. Add search endpoint which calculates cosine similarity (dot product) against potential relevant values
5. Add all vectorized data
6. Let your users search

## Potential Strategies

## Sparse Retrieval/Key word search

### How to build a sparse retrieval system for text

TF-IDF (Term Frequency-Inverse Document Frequency) and BM-25 are both keyword-based retrieval models used to rank documents by relevance to a query. Here’s a step-by-step completion and correction of the process:

1. Tokenization: For each document, tokenize the text, which involves breaking the text into words, phrases, or other meaningful elements called tokens.
2. Create a Vocabulary: Construct a list of unique words from all the documents, which forms the vocabulary.
3. Vectorization:

   • For TF-IDF: Create a vector for each document that counts the frequency of each word in the vocabulary within that document. Adjust these counts based on the rarity of the word across all documents. This gives high values to words that are frequent in a document but not common in the overall corpus.

   • For BM-25: Similar to TF-IDF, but it includes additional factors like the average document length in the corpus to adjust the term frequency component, making it more sophisticated and often more effective.

4. Query Processing: When a query is received, it is also tokenized and transformed into a vector using the same method as the documents.
5. Ranking:

   • For TF-IDF: Calculate the cosine similarity between the TF-IDF vector of the query and the TF-IDF vectors of each document. The documents are then ranked based on their similarity scores.

   • For BM-25: Apply the BM-25 formula which uses the term frequency (TF), inverse document frequency (IDF), and the length of documents to score each document relative to the query. Documents are ranked based on these scores.

6. Retrieval: The top-ranked documents are then retrieved and presented as the results for the query.

Both TF-IDF and BM-25 fundamentally rely on the idea that if a word appears frequently in a document but not too frequently across all documents, it is likely important for understanding the content of that document, making the document more relevant to queries containing that word.

### How to apply sparse techniques to images

You'll notice that this is an extra step required for sparse techniques, as there isn't a sparse representation for images, as sparse techniques are just applying basics from NLP.

Therefore, we have to get some natural language representation of the image, which fundamentally will not be as good as a model jointly trained on both images and text for your use case. That is a strong argument for dense retrieval methods.

1. For any give image, collect any associated text
2. Run the image through a Vision Language Model (VLM) that describes the image with some prompt relevant to the use case.
3. Associate the image with the associated blocks of text from the VLM and your own metadata.
4. Vectorize and store only the vectors from the text.
5. Just do cosine similarity with a dot product of your query against potential relevant vectors.

### When to use sparse retrieval

Embedding based retrieval is the sexy new hotness and is discussed in the next section. But it is worth looking at the really simple sparse retrieval methods and seeing if that can make more sense for whole collection of reasons.

1. Vectors for embeddings are big and to be helpful, normally target fairly small blocks of text. This means that the size of your vectors can end up being as large as the data you are searching.
2. Ya probably need a GPU for embeddings. Calculating vectors on a CPU is slow, so you need to use a GPU for most use cases. If you are ingesting a lot of data that you want to be searchable quickly, all of that data needs to be sent (probably over the network) to the GPU so vectors can be calculated, then when the user does search the dot product is also ideally then done on a GPU (although that is less important.) This means storing large amounts of additional data, large amounts of additional network load.
3. For many use cases, embeddings aren't significantly better then sparse retrieval techniques like BM-25. For code retrieval, the performance difference is fairly marginal. Citation needed, but here is the literature review I wrote showing 66% vs 68% successful retrieval for copilot esque systems: https://arxiv.org/pdf/2312.10101
4. Because you do not have to manage vectors sparse retrieval systems are normally much simpler to implement. See Bleve: https://github.com/blevesearch/bleve or it's cousin Bluge: https://github.com/blugelabs/bluge, but TF-IDF and even BM-25 are fairly standard algos at this point
5. Your users likely will just use keywords anyway! These often times are not actually as semantically close to the actual data you want when the semantic representations are based off of full sentences rather than keywords.
6. Like with most ML tasks, the real solution is an ensemble model. Combining sparse vector and dense vector based retrieval often benchmarks the highest.

## Embeddings/dense vector powered retrieval

### Get an embedding model

There are a couple options here:

1. Finetune an embedding model, for this I am pointing to [CLIP](https://github.com/openai/CLIP) from OpenAI, because it is good, easy to use, and open source. It is from 2021 so it likely isn't SoTA anymore. You can see some rough python showing approximately how to do this in Train.py

2. Alternatively, you can build an embedding model completely from scratch. Given the emergent properties we appear to see from generalized scale, I would be extremely surprised if this was worth doing, unless you have an ungodly amount of data both generalized and specialized.

3. Use a specialized existing embedding model out of the box. If you are working on something like embeddings for code, this likely will work very well because a lot of the low hanging fruit will already have been optimized for you by hundreds of researchers. If you are working on say manufacturing defect recognition and text analysis, it is likely there isn't a massive amount of that in the existing embedding models dataset and therefore finetuning will be worth doing and there is unlikely to be an existing model for that use case.

### Finetune the embedding model

- See this python script for the naive set up on how to do this. In practice you need some scalable infra to be able to rapidly process and serve your data. It can be easy to make querying shuffled data slower than the GPU is doing training on it. (Found this out while doing seizure prediction stuff)
- This really requires an entire machine learning data pipeline so you can mitigate data drift/keep your models up to date in your changing org etc. It is a non-trivial problem.

## Build search infra

The naive way to do this is to dump all your vectors in a numpy array and to load them and when you get a search query you just dot product against them, and then return the top k values making sure to keep these vectors associated with the true original text.

Things start getting more interesting when you start putting everything in a graph so you are querying a graph database and doing things like pagerank to bump up the neighbors of high ranking results and to bump down neighbors of low ranking results.

For example, if you submit a query, "widget 6 cracks" you might have a graph where widget 5 and 6 are connected to different sub-components. Maybe a large collection of widget 5 results show up naively, but then the widget 6 crack results are upscaled more than the widget 5 results are due to being a neighbor of another more relevant search term (widget 6 vs widget 5)

Eventually, you will need to shard your vectors and do queries in parallel. You'd need A LOT of vectors though.

# How do I use AI with it?

Host llama 70B with [VLLM](https://github.com/vllm-project/vllm) or [tensor-RT](https://docs.nvidia.com/deeplearning/tensorrt/) on your GPUs.

VLLM doesn't do as much quantization and model optimization for your hardware, but is easier to set up.

Then just have the model generate a search query or just run the users question automatically as a search query and then feed the results to your model.

It is worth noting that there aren't a lot of good open source models that are multi-modal.

But no matter who you are you should be able to run GPT-4 on an azure endpoint:
https://learn.microsoft.com/en-us/legal/cognitive-services/openai/data-privacy#how-does-the-azure-openai-service-process-data

The only tricky bit here would be you may have to negotiate with azure if you do not want your data crossing into international regions or to be processed by their content abuse filters. However, if your organization is large enough that should be perfectly tractable.

Another option is AWS bedrock.
Unlike azure, they actually do not say the models are stateless and claim your data is stored in your aws infra, otherwise they have reasonable security standards depending on your orgs needs.
https://aws.amazon.com/bedrock/faqs/
