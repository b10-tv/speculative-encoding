"""Speculative encoding: translate a source LLM's document states into a target LLM's KV
cache, optionally compress it per question, and answer from it with vLLM or Hugging Face.

Start with ``speculative_encoding.api`` (``SpeculativeEncoder``, ``MultiTargetEncoder``)
or the command line, ``python -m speculative_encoding.cli``."""
