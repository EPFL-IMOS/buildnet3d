import sys
import tyro
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

sys.path.append(".")
from nerfstudio.scripts.train import main
from buildnet3d.models.method_configs import (
    NeRFactoTrackConfig,
    NeuSTrackConfig,
    SemanticSDFTrackConfig
)

@dataclass
class TrainingConfig:
    """Configuration for training"""

    model_type: str = "semantic-sdf"
    """What NeRF model to train. Defaults to NeuS"""
    experiment_name: str = "neus_train"
    """Name of the model to train"""
    output_dir: Path = "./outputs"
    """Where to save the model and outputs"""
    max_num_iterations: int = 100001
    data: Path = "./inputs"
    """Where to find the input data """
    load_config: Optional[Path] = None
    load_dir: Optional[Path] = None
    
    use_appearance_embedding: Optional[bool] = False
    use_average_appearance_embedding: Optional[bool] = True
    use_grid_feature: Optional[bool] = True
    use_transient_embedding: Optional[bool] = False
    
    def __post_init__(self) -> None:
        method_config = {
            "nerfacto": NeRFactoTrackConfig,
            "neus": NeuSTrackConfig,
            "semantic-sdf": SemanticSDFTrackConfig,
        }
        if self.model_type not in method_config:
            raise ValueError(f"Model type {self.model_type} not supported.")
        self.model = method_config[self.model_type]
        self.model.experiment_name = self.experiment_name
        self.model.output_dir = self.output_dir
        self.model.max_num_iterations = self.max_num_iterations
        self.model.data = self.data
        self.model.load_config = self.load_config
        self.model.load_dir = self.load_dir
        
        if self.model_type in ['neus', "semantic-sdf"]:
            self.model.pipeline.model.sdf_field.use_appearance_embedding = self.use_appearance_embedding
            self.model.pipeline.model.sdf_field.use_grid_feature = self.use_grid_feature
            self.model.pipeline.model.use_average_appearance_embedding = self.use_average_appearance_embedding


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)
    
    main(config.model)
