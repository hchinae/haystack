# ## Task: Build a Question Answering pipeline without Elasticsearch
#
# Haystack provides alternatives to Elasticsearch for developing quick prototypes.
#
# You can use an `InMemoryDocumentStore` or a `SQLDocumentStore`(with SQLite) as the document store.
#
# If you are interested in more feature-rich Elasticsearch, then please refer to the Tutorial 1.

#from curses.ascii import EM
from pprint import pprint
from statistics import mode
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import InMemoryDocumentStore, SQLDocumentStore, ElasticsearchDocumentStore
from haystack.nodes import FARMReader, TransformersReader, TfidfRetriever, ElasticsearchRetriever, retriever, EmbeddingRetriever
from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http, print_answers, launch_es


from haystack.document_stores import PineconeDocumentStore

from haystack.nodes import DensePassageRetriever

def qa_pipeline_PineconeDocumentStore():
    
    # In-Memory Document Store
    document_store = InMemoryDocumentStore()
    
    # ## Preprocessing of documents
    #
    # Haystack provides a customizable pipeline for:
    # - converting files into texts
    # - cleaning texts
    # - splitting texts
    # - writing them to a Document Store

    # Let's first get some documents that we want to query
    # Here: RichDadPoorDad
    doc_dir = "books/"
    # convert files to dicts containing documents that can be indexed to our datastore
    docs = convert_files_to_docs(
        dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
    # You can optionally supply a cleaning function that is applied to each doc (e.g. to remove footers)
    # It must take a str as input, and return a str.

    
    document_store.write_documents(docs)

    # ## Initalize Retriever, Reader & Pipeline
    #
    # ### Retriever
    #
    # Retrievers help narrowing down the scope for the Reader to smaller units of text where
    # a given question could be answered.
    #
    # With InMemoryDocumentStore or SQLDocumentStore, you can use the TfidfRetriever. For more
    # retrievers, please refer to the tutorial-1.

    retriever = TfidfRetriever(document_store=document_store)
    


    contexts = retriever.retrieve("What is the importance of accounting?") 
    print(contexts[0].content)
    

    # ### Reader
    #
    # A Reader scans the texts returned by retrievers in detail and extracts the k best answers. They are based
    # on powerful, but slower deep learning models.
    #
    # Haystack currently supports Readers based on the frameworks FARM and Transformers.
    # With both you can either load a local model or one from Hugging Face's model hub (https://huggingface.co/models).

    # **Here:**                   a medium sized RoBERTa QA model using a Reader based on
    #                             FARM (https://huggingface.co/deepset/roberta-base-squad2)
    # **Alternatives (Reader):**  TransformersReader (leveraging the `pipeline` of the Transformers package)
    # **Alternatives (Models):**  e.g. "distilbert-base-uncased-distilled-squad" (fast) or
    #                             "deepset/bert-large-uncased-whole-word-masking-squad2" (good accuracy)
    # **Hint:**                   You can adjust the model to return "no answer possible" with the no_ans_boost.
    #                             Higher values mean the model prefers "no answer possible".


    # #### FARMReader
    #
    # Load a  local model or any of the QA models on
    # Hugging Face's model hub (https://huggingface.co/models)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    # #### TransformersReader
    # Alternative:
    # reader = TransformersReader(model_name_or_path="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)

    # ### Pipeline
    #
    # With a Haystack `Pipeline` you can stick together your building blocks to a search pipeline.
    # Under the hood, `Pipelines` are Directed Acyclic Graphs (DAGs) that you can easily customize for your own use cases.
    # To speed things up, Haystack also comes with a few predefined Pipelines. One of them is the `ExtractiveQAPipeline` that combines a retriever and a reader to answer our questions.
    # You can learn more about `Pipelines` in the [docs](https://haystack.deepset.ai/docs/latest/pipelinesmd).

    pipe = ExtractiveQAPipeline(reader, retriever)

    # Voilà! Ask a question!
    prediction = pipe.run(
        query="What is the importance of accounting?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}}
    )

    # prediction = pipe.run(query="Who created the Dothraki vocabulary?", params={"Reader": {"top_k": 5}})
    # prediction = pipe.run(query="Who is the sister of Sansa?", params={"Reader": {"top_k": 5}})

    # Now you can either print the object directly
    print("\n\nRaw object:\n")

    pprint(prediction)

    
    # Or use a util to simplify the output
    # Change `minimum` to `medium` or `all` to raise the level of detail
    print("\n\nSimplified output:\n")
    print_answers(prediction, details="minimum")




if __name__ == "__main__":
    qa_pipeline_PineconeDocumentStore()

# This Haystack script was made with love by deepset in Berlin, Germany
# Haystack: https://github.com/deepset-ai/haystack
# deepset: https://deepset.ai/
