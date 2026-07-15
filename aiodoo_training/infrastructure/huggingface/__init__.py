"""HuggingFace infrastructure adapters."""

from aiodoo_training.infrastructure.huggingface.checkpoint_store import (
    HFCheckpointStore,
    register_hf_checkpoint_store,
)
from aiodoo_training.infrastructure.huggingface.evaluator import HFEvaluator, register_hf_evaluator
from aiodoo_training.infrastructure.huggingface.exporter import HFExporter, register_hf_exporter
from aiodoo_training.infrastructure.huggingface.model import (
    HuggingFaceCausalLMBackend,
    register_default_model_backends,
)
from aiodoo_training.infrastructure.huggingface.stub_model import StubModelBackend
from aiodoo_training.infrastructure.huggingface.templates import (
    register_default_chat_templates,
)
from aiodoo_training.infrastructure.huggingface.tokenizer import (
    DeterministicStubTokenizer,
    HuggingFaceTokenizerAdapter,
    register_default_tokenizers,
)
from aiodoo_training.infrastructure.huggingface.trainer import HFTrainerBackend, register_hf_trainer

__all__ = [
    "DeterministicStubTokenizer",
    "HFEvaluator",
    "HFExporter",
    "HFCheckpointStore",
    "HFTrainerBackend",
    "HuggingFaceCausalLMBackend",
    "HuggingFaceTokenizerAdapter",
    "StubModelBackend",
    "register_default_chat_templates",
    "register_default_model_backends",
    "register_default_tokenizers",
    "register_hf_checkpoint_store",
    "register_hf_evaluator",
    "register_hf_exporter",
    "register_hf_trainer",
]
