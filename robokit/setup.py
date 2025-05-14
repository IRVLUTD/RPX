#----------------------------------------------------------------------------------------------------
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.
# 🖋️ Jishnu Jaykumar Padalunkal (2025).
#----------------------------------------------------------------------------------------------------

import os
import requests
import subprocess
import setuptools
from tqdm import tqdm
from setuptools.command.install import install
import logging
from absl import app

class FileFetch(install):
    def run(self):
        install.run(self)
        robokit_root_dir = os.getcwd()

        # Install required GitHub dependencies
        subprocess.run([
            "pip", "install", "-U",
            'git+https://github.com/IDEA-Research/GroundingDINO.git@2b62f419c292ca9c518daae55512fabc3fead4a4',
            'git+https://github.com/ChaoningZhang/MobileSAM@c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed',
        ])

        # Clone and install SAM2
        samv2_dir = os.path.join(robokit_root_dir, "robokit", "sam2")
        os.makedirs(samv2_dir, exist_ok=True)
        try:
            subprocess.run(["git", "clone", "https://github.com/facebookresearch/sam2", samv2_dir], check=True)
        except:
            pass
        os.chdir(samv2_dir)
        subprocess.run(["git", "checkout", "--branch", "c2ec8e14a185632b0a5d8b161928ceb50197eddc"])
        subprocess.run(["sed", "-i", "171s/^/#/", "setup.py"], check=True)
        subprocess.run(["python", "setup.py", "install"], check=True)
        os.chdir(robokit_root_dir)

        # Download GroundingDINO checkpoint
        self.download_pytorch_checkpoint(
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            os.path.join(os.getcwd(), "ckpts", "gdino"),
            "gdino.pth"
        )

        # Download MobileSAM checkpoint
        self.download_pytorch_checkpoint(
            "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
            os.path.join(os.getcwd(), "ckpts", "mobilesam"),
            "vit_t.pth"
        )

        # Download SAM2 checkpoint and config
        self.download_pytorch_checkpoint(
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
            os.path.join(os.getcwd(), "ckpts", "samv2"),
            "sam2.1_hiera_large.pth"
        )
        self.download_pytorch_checkpoint(
            "https://raw.githubusercontent.com/facebookresearch/sam2/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            os.path.join(os.getcwd(), "ckpts", "samv2"),
            "sam2.1_hiera_l.yaml"
        )

    def download_pytorch_checkpoint(self, pth_url: str, save_path: str, renamed_file: str):
        try:
            file_path = os.path.join(save_path, renamed_file)
            if os.path.exists(file_path):
                logging.info(f"{file_path} already exists! Skipping download")
                return

            os.makedirs(save_path, exist_ok=True)
            logging.info("Attempting to download PyTorch checkpoint from: %s", pth_url)

            response = requests.get(pth_url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

            with open(file_path, 'wb') as file:
                for data in response.iter_content(chunk_size=block_size):
                    progress_bar.update(len(data))
                    file.write(data)

            progress_bar.close()
            logging.info("Checkpoint downloaded and saved to: %s", file_path)

        except Exception as e:
            logging.error("Error downloading checkpoint: %s", e)
            raise e

def run_setup(argv):
    del argv
    with open('requirements.txt', 'r') as f:
        requirements = f.read().splitlines()

    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setuptools.setup(
        name="RoboKit",
        version="0.0.1",
        author="Jishnu P",
        author_email="jishnu.p@utdallas.edu",
        description="A toolkit for robotic tasks",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/IRVLUTD/RoboKit",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        package_dir={"": "robokit"},
        packages=setuptools.find_packages(where="robokit"),
        python_requires=">=3.0",
        install_requires=requirements,
        cmdclass={
            'install': FileFetch,
        },
        package_data={'gdino_cfg': ["robokit/cfg/gdino/GroundingDINO_SwinT_OGC.py"]}
    )

if __name__ == "__main__":
    app.run(run_setup)
