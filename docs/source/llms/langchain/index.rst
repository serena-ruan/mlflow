MLflow LangChain Flavor
==========================

.. attention::
    The ``langchain`` flavor is in active development and is marked as Experimental. Public APIs may change and new features are
    subject to be added as additional functionality is brought to the flavor.

Introduction
------------

LangChain is a framework for building applications powered by language models. It makes it easy to connect language models to 
different sources of context like prompt instructions, examples, and content to ground the model's responses. 
LangChain also simplifies building chains of reasoning where a language model chooses which tools or functions 
to call to accomplish some task.

The LangChain framework has Python and JavaScript packages with interfaces and integrations for components like 
language models, databases, file systems etc. It also comes with pre-built chain implementations for common use cases. 
Developers can either use these off-the-shelf chains directly or customize them by mixing and matching components.

Some benefits of LangChain are:

* Modular components that are easy to combine
* Reference architectures and templates to accelerate development
* Tools like LangSmith to debug, test and monitor chains
* Easy deployment of chains via LangServe by turning them into APIs

Some key components of LangChain are:

* LLMs (Language Models): The core reasoning engine. LangChain supports both standard language models (LLMs) that take 
string input and return string output, as well as conversational models (ChatModels) that take a list of messages 
as input and return a message.
* Prompt Templates: Used to format user input into a prompt for the LLM. Allow variables to be inserted into a 
template. Can be composed together.
* Output Parsers: Convert raw LLM output into a usable format. For example, parsing text into JSON or just 
extracting a string from a ChatMessage.
* Chains: Combine a prompt template, LLM, and output parser into a modular piece of logic. 
Take input variables, format into a prompt, call LLM, and parse output.

In summary, LangChain provides building blocks like LLMs, prompts, parsers and chains that can be used to create 
language model powered applications. The components interface with each other and allow customization at each level.

**MLflow's Langchain Flavor**: 

The ``langchain`` model flavor enables logging of `LangChain models <https://github.com/hwchase17/langchain>`_ in MLflow format via
the :py:func:`mlflow.langchain.save_model()` and :py:func:`mlflow.langchain.log_model()` functions. Use of these
functions also adds the ``python_function`` flavor to the MLflow Models that they produce, allowing the model to be
interpreted as a generic Python function for inference via :py:func:`mlflow.pyfunc.load_model()`.
You can also use the :py:func:`mlflow.langchain.load_model()` function to load a saved or logged MLflow
Model with the ``langchain`` flavor as a dictionary of the model's attributes.

MLflow is natively integrated with `Langchain <https://python.langchain.com/docs/integrations/providers/mlflow_tracking>`_., 
allowing you to log and load Langchain models as MLflow Models.


Logging RetrievalQA Chains
~~~~~~~~~~~~~~~~~~~~~~~~~~

In MLflow, you can use the ``langchain`` flavor to save a ``RetrievalQA`` chain, including the retriever object.

Native LangChain requires the user to handle the serialization and deserialization of the retriever object, but MLflow's ``langchain`` flavor handles that for you.

Here are the two things you need to tell MLflow:

1. Where the retriever object is stored (``persist_dir``).
2. How to load the retriever object from that location (``loader_fn``).

After you define these, MLflow takes care of the rest, saving both the content in the ``persist_dir`` and pickling the ``loader_fn`` function.

Example: Log a LangChain RetrievalQA Chain

.. literalinclude:: ../../examples/langchain/retrieval_qa_chain.py
    :language: python

.. code-block:: python
    :caption: Output (truncated)

    [" The president said..."]

.. _log-retriever-chain:

Logging a retriever and evaluate it individually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``langchain`` flavor provides the functionality to log a retriever object and evaluate it individually. This is useful if
you want to evaluate the quality of the relevant documents returned by a retriever object without directing these documents
through a large language model (LLM) to yield a summarized response.

In order to log the retriever object in the ``langchain`` flavor, it is also required to specify ``persist_dir``
and ``loader_fn``, the same as logging the RetrievalQA chain. See the previous section for details about these parameters.

See the following example for more details.

Example: Log a LangChain Retriever

.. literalinclude:: ../../examples/langchain/retriever_chain.py
    :language: python

.. code-block:: python
    :caption: Output (truncated)

    [
        [
            {
                "page_content": "Tonight. I call...",
                "metadata": {"source": "/state.txt"},
            },
            {
                "page_content": "A former top...",
                "metadata": {"source": "/state.txt"},
            },
        ]
    ]


Getting Started with the MLflow Langchain Flavor - Tutorials and Guides
--------------------------------------------------------------------------

Below, you will find a number of guides that focus on different use cases (`tasks`) using `langchain`  that leverage MLflow's 
APIs for tracking and inference capabilities. 

Introductory Quickstart to using Langchain with MLflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If this is your first exposure to langchain or use langchain extensively but are new to MLflow, this is a great place to start.

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="tutorials/llm-query/simple_chain.html">
                    <div class="header">
                        Quickstart: Use OpenAI with langchain and MLflow
                    </div>
                    <p>
                        Learn how to leverage the langchain integration with MLflow in this <strong>introductory quickstart</strong>.
                    </p>
                </a>
            </div>
        </article>
    </section>

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/llms/langchain/tutorials/llm-query/simple_chain.ipynb" class="notebook-download-btn">Download the Introductory LLM Chain Notebook</a><br>
    
Use Case Tutorials for Langchain with MLflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interested in learning about how to leverage langchain for tasks other than chat completion? Want to learn more about the breadth of problems that you can solve with langchain and MLflow? 

These more advanced tutorials are designed to showcase different applications of the langchain model architecture and how to leverage MLflow to track and deploy these models.

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="tutorials/agents/simple_agent.html">
                    <div class="header">
                        Langchain Agents
                    </div>
                    <p>
                        Learn how to use langchain agents with MLflow to choose a sequence of actions to take.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="tutorials/document-retrieval/retriever_chain.html">
                    <div class="header">
                        Retrieve documents powered by vector store with Langchain
                    </div>
                    <p>
                        Learn how to use vector store retriever in MLflow for retrieving documents!
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="tutorials/question-answering/retrieval_qa_chain.html">
                    <div class="header">
                        Question answering with Langchain
                    </div>
                    <p>
                        Learn how to build a basic RAG with LangChain and MLflow.
                    </p>
                </a>
            </div>
        </article>
    </section>