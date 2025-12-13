.. pytest-texts-score documentation master file, created by
   sphinx-quickstart on Sat Dec 13 19:58:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pytest-texts-score documentation
================================

A **pytest plugin for semantic text similarity scoring** using Large Language Models (LLMs).

It enables robust assertions over *meaning*, not surface text, making it ideal for validating
LLM outputs, RAG systems, summaries, and other generated content.

The plugin evaluates similarity by prompting an LLM to extract and answer factual questions,
producing **Precision (Completeness)**, **Recall (Correctness)**, and **F1** scores.


Metrics overview
~~~~~~~~~~~~~~~~

• **Recall (Correctness)**  
  Measures how much information from the *expected* text is present in the *given* text.

• **Precision (Completeness)**  
  Measures how much information in the *given* text is supported by the *expected* text.

• **F1 score**  
  Harmonic mean of precision and recall.


Aggregated assertions
~~~~~~~~~~~~~~~~~~~~~

These perform **multiple evaluations** and aggregate the result.
Recommended for CI/CD pipelines to reduce LLM nondeterminism.

Supported aggregations: ``min``, ``max``, ``median``, ``mean`` / ``average``.


----


pytest\_texts\_score package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pytest_texts_score
   :members:
   :show-inheritance:
   :undoc-members:
   :noindex:



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
