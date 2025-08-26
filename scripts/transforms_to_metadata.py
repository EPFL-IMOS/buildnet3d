import json
from pathlib import Path

def transforms_to_metadata(transforms_path: str, metadata_path: str, add_semantics: bool = False) -> None:
    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    # Build metadata
    metadata = {
        "camera_model": "OPEN_CV",  # rename
        "width": transforms["w"],
        "height": transforms["h"],
        "k1": transforms["k1"],
        "k2": transforms["k2"],
        "p1": transforms["p1"],
        "p2": transforms["p2"],
        "has_mono_prior": False,
        "has_foreground_mask": False,
        "has_sparse_sfm_points": False,
        "scene_box": {
            "aabb": [[-1, -1, -1], [1, 1, 1]]
        },
        "frames": []
    }

    # Construct frames
    for frame in transforms["frames"]:
        filename = Path(frame["file_path"]).name
        intrinsics = [
            [transforms["fl_x"], 0, transforms["cx"]],
            [0, transforms["fl_y"], transforms["cy"]],
            [0, 0, 1],
        ]

        metadata["frames"].append({
            "rgb_path": filename,
            "camtoworld": frame["transform_matrix"],
            "intrinsics": intrinsics
        })
        
        print(filename)
        if add_semantics:
            metadata["frames"][-1]["segmentation_path"] = filename

    # Write output
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Saved {metadata_path}")


if __name__ == "__main__":
    transforms_to_metadata("/home/chexu/data/transforms.json", "/home/chexu/data/meta_data.json")
