## Multi-Modal Imageset Generation

This module provides a `BlenderProc`-based pipeline for rendering **multi-modal image data** from 3D building models. It is designed to support research and applications in vision-based scene reconstruction and understanding, particularly in structured environments such as building envelopes.

### Key Features

- **Multi-View Rendering**  
  Ensures spatial consistency across multiple views to support downstream tasks such as 3D reconstruction and semantic scene understanding.

- **Multi-Modal Outputs**  
  Generates ground-truth image data for each view, including:
  - RGB images  
  - Depth maps  
  - Surface normal maps  
  - Semantic segmentation labels  
  - Instance segmentation masks  

- **Rule-Based Camera Sampling**  
  Automatically samples camera viewpoints using a simple yet effective rule-based strategy, eliminating the need to manually define large sets of camera poses. The algorithm ensures full object visibility in each frame.

- **Customizable Rendering Parameters**  
  The rendering behavior is configurable via the `RenderParams()` class. Users can easily adjust control parameters, specify custom 3D models, and configure background settings to suit specific use cases.


### Usage

```bash
# Install BlenderProc
pip install blenderproc

# Additional dependencies within the BlenderProc environment
blenderproc pip install tyro

# Run the image generation pipeline
blenderproc run bproc_generator/render/generate.py
```
**Note**: To enable semantic/instance segmentation outputs, input 3D models must be properly segmented and loaded.
You can find an example 3D building model to test our framework [here]().