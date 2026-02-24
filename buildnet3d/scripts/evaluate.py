import tyro
import subprocess
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TestingConfig:
    """Configuration for evaluation"""
    load_config: Path
    """Path to the trained run config.yml"""
    output_path: Path = Path("./eval.json")
    """Where to save evaluation metrics JSON"""
    render_output_path: Optional[Path] = None
    """Optional directory to save rendered eval images"""
    project_root: Path = Path(__file__).resolve().parents[2]
    """Project root to add to PYTHONPATH (should contain buildnet3d/)"""

    def run(self) -> None:
        cmd = [
            "ns-eval",
            "--load-config",
            str(self.load_config),
            "--output-path",
            str(self.output_path),
        ]
        if self.render_output_path is not None:
            cmd.extend(["--render-output-path", str(self.render_output_path)])

        env = os.environ.copy()
        old_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{self.project_root}:{old_pythonpath}" if old_pythonpath else str(self.project_root)
        )
        subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    config = tyro.cli(TestingConfig)
    config.run()