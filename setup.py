from setuptools import setup, find_packages

setup(
    name="unify-diffusion",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "datasets",
        "huggingface_hub",
        "lightning",
        "einops",
        "hydra-core",
        "omegaconf",
        "wandb",
        "transformers",
        "faiss-cpu",
        "pytorch-lightning",
        "evodiff",
        "numba",
        "fair-esm",
        "h5py",
    ],
    python_requires='>=3.8.5',
)
