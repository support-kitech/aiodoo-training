"""Port ABC surface tests — ensure ports remain abstract."""

import pytest

from aiodoo_training.ports import (
    AdaptationStrategy,
    ChatTemplate,
    CheckpointStore,
    CurriculumStrategy,
    DatasetSource,
    Evaluator,
    ExampleFormatter,
    ExperimentTracker,
    Exporter,
    ModelBackend,
    PackingStrategy,
    ResourcePlanner,
    RngController,
    TokenizerPort,
    TrainerBackend,
)


@pytest.mark.parametrize(
    "port",
    [
        DatasetSource,
        ExampleFormatter,
        TokenizerPort,
        ChatTemplate,
        ModelBackend,
        AdaptationStrategy,
        PackingStrategy,
        CurriculumStrategy,
        TrainerBackend,
        CheckpointStore,
        Evaluator,
        Exporter,
        ExperimentTracker,
        RngController,
        ResourcePlanner,
    ],
)
def test_ports_are_abstract(port: type) -> None:
    with pytest.raises(TypeError):
        port()  # type: ignore[call-arg]
