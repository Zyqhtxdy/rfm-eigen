from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    method: str
    seed: int
    device: str
    dtype: str
    width: int
    outputs: int
    learning_rate: float
    paper_epochs: int
    max_epochs: int
    batch_size: int
    boundary_batch_size: int
    eval_every: int
    checkpoint_every: int
    minimum_learning_rate: float
    plateau_patience_evals: int
    stop_patience_epochs: int
    result_name: str
    interface_batch_size: int = 4096
    training_track: str = "source_faithful"
    sampling_mode: str = "full_grid"
    collocation_nx: int = 171
    points_per_cell: int = 10
    validation_nx: int = 86
    validation_points_per_cell: int = 5
    validation_interior_points: int = 4096
    scheduler_enabled: bool = False
    allow_early_stopping: bool = False

    @classmethod
    def from_toml(cls, path: str | Path) -> RunConfig:
        with Path(path).open("rb") as handle:
            values = tomllib.load(handle)
        return cls(**values)

    def validate(self) -> None:
        if self.method not in {"drm", "gipmnn", "pc_gipmnn"}:
            raise ValueError(f"Unsupported method: {self.method}")
        expected_outputs = 7 if self.method == "pc_gipmnn" else 1
        if self.outputs != expected_outputs:
            raise ValueError(f"{self.method} requires outputs={expected_outputs}")
        if self.paper_epochs > self.max_epochs:
            raise ValueError("paper_epochs cannot exceed max_epochs")
        if self.training_track not in {"source_faithful", "extended_to_convergence"}:
            raise ValueError(
                "training_track must be 'source_faithful' or 'extended_to_convergence'"
            )
        if self.sampling_mode != "full_grid":
            raise ValueError(
                "Example 4 formal runs require sampling_mode='full_grid'; "
                "minibatch results are not admissible for the source-paper comparison."
            )
        if any(value != 0 for value in (
            self.batch_size,
            self.boundary_batch_size,
            self.interface_batch_size,
        )):
            raise ValueError(
                "Full-grid mode uses every interior, boundary, and interface point; "
                "set all three batch-size fields to 0 (the explicit sentinel for all points)."
            )
        if self.collocation_nx != 171:
            raise ValueError("Example 4 source-faithful data require collocation_nx=171")
        if self.validation_nx == self.collocation_nx:
            raise ValueError("The fixed validation grid must differ from the training grid")
        if self.validation_interior_points <= 0:
            raise ValueError("validation_interior_points must be positive")
        if self.training_track == "source_faithful":
            if self.max_epochs != self.paper_epochs:
                raise ValueError("source_faithful runs stop exactly at the source-paper budget")
            if self.scheduler_enabled or self.allow_early_stopping:
                raise ValueError(
                    "source_faithful runs use the fixed source budget without adaptive scheduling/early stop"
                )
        elif self.max_epochs <= self.paper_epochs:
            raise ValueError("extended_to_convergence requires max_epochs > paper_epochs")
