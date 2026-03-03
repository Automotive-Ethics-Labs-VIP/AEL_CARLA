from setuptools import setup

setup(
    name="carla-ethical-sim",
    version="0.1.0",
    description="Ethical pedestrian attribute layer for CARLA simulations",
    python_requires=">=3.8",
    packages=["python_api", "python_api.stream"],
    install_requires=[
        "pydantic>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=9.0",
        ],
    },
)