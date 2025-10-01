from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig
)

from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
    CosineDecaySchedulerConfig
    )

from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.neus import NeuSModelConfig
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

from buildnet3d.data.buildnet_dataset import BuildNetDataset
from buildnet3d.data.buildnet_dataparser import BuildNetDataParserConfig

from buildnet3d.models.semantic_sdf import SemanticSDFModelConfig
from buildnet3d.models.semantic_sdf_field import SemanticSDFFieldConfig

NeuSTrackConfig = TrainerConfig(
    method_name="neus",
    steps_per_eval_image=1000,
    steps_per_eval_batch=5000,
    steps_per_save=10000,
    steps_per_eval_all_images=1000000,
    max_num_iterations=300001,
    save_only_latest_checkpoint= False,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[BuildNetDataset],
            dataparser=BuildNetDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
        ),
        model=NeuSModelConfig(
            sdf_field=SDFFieldConfig(
                bias=0.5,
                beta_init=0.2,
                inside_outside=False,
            ),
            near_plane=0.01,
            far_plane=50.0,
            background_model="none",
            overwrite_near_far_plane=True,
            eval_num_rays_per_chunk=1024,
        ),
    ),
    optimizers={
        "camera_opt": {
            "mode": "off",
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=10000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(
                warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300001
            ),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(
                warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300001
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="wandb",
)

NeRFactoTrackConfig = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_batch=5000,
    steps_per_save=5000,
    max_num_iterations=50001,
    save_only_latest_checkpoint= False,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BuildNetDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
        ),
        model=NerfactoModelConfig(
            eval_num_rays_per_chunk=1 << 15,
            average_init_density=0.01,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=100000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=100000),
        },
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=10000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
    vis="wandb",
)

SemanticSDFTrackConfig = TrainerConfig(
    method_name="semantic-sdf",
    steps_per_eval_image=1000,
    steps_per_eval_batch=5000,
    steps_per_save=10000,
    steps_per_eval_all_images=1000000,
    max_num_iterations=300001,
    save_only_latest_checkpoint= True,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[BuildNetDataset],
            dataparser=BuildNetDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
        ),
        model=SemanticSDFModelConfig(
            near_plane=0.5,
            far_plane=50.0,
            overwrite_near_far_plane=False,
            sdf_field=SemanticSDFFieldConfig(
                num_layers=8,
                num_layers_color=4,
                hidden_dim=256,
                bias=0.5,
                beta_init=0.2,
                inside_outside=False,
            ),
            background_model="none",
            eval_num_rays_per_chunk=1024,
            semantic_loss_mult=0.5,
            eikonal_loss_mult=0.1,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(
                warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300001
            ),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(
                warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300001
            ),
        "camera_opt": {
            "mode": "off",
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=10000),
        },
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="wandb",
)
