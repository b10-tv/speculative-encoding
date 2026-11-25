"""Run the pipeline on a Baseten GPU (examples/baseten_run.sh): the released checkpoint, one document, one question,
answered with vLLM; the answers are in the job's logs.

    HF_SECRET=<your Baseten secret holding a Hugging Face token> \\
      uvx truss train push run_on_baseten.py --job-name speculative-encoding-demo

The token needs read access to the gated catalog models (Llama 3.1, Gemma). Environment variables at push time are
passed to the job: SOURCE, TARGET, DOCUMENT, DOCUMENT_URL, DOCUMENT_CHARS, QUESTION, COMPRESSOR (see examples/baseten_run.sh);
ACCELERATOR (default H100). ``truss train push`` ships this folder (minus .truss_ignore)."""

from __future__ import annotations

import os

from truss.base.truss_config import AcceleratorSpec
from truss_train import Compute, Image, Runtime, SecretReference, TrainingJob, TrainingProject, definitions

PASSED = ("SOURCE", "TARGET", "DOCUMENT", "DOCUMENT_URL", "DOCUMENT_CHARS", "QUESTION", "COMPRESSOR")

training_project = TrainingProject(
    name=os.environ.get("PROJECT", "speculative-encoding"),
    job=TrainingJob(
        image=Image(base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel"),  # Python and nvcc; uv brings Torch 2.11
        compute=Compute(
            node_count=1,
            accelerator=AcceleratorSpec(accelerator=os.environ.get("ACCELERATOR", "H100"), count=1),
        ),
        runtime=Runtime(
            start_commands=["bash examples/baseten_run.sh"],
            environment_variables={
                "HF_TOKEN": SecretReference(name=os.environ["HF_SECRET"]),
                **{key: os.environ[key] for key in PASSED if os.environ.get(key)},
            },
            cache_config=definitions.CacheConfig(enabled=False),
        ),
    ),
)
