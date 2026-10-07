from setuptools import setup

setup(
    name="carla-ethical-sim",
    version="0.1.0",
    description="Ethical pedestrian attribute layer for CARLA simulations",
    python_requires=">=3.8",
    packages=["python_api", "python_api.stream"],
    install_requires=[
        "pydantic>=2.0",
        "ael-common @ git+https://github.com/Automotive-Ethics-Labs-VIP/ael-common@v0.1.1",
    ],
    extras_require={
        "dev": [
            "pytest>=9.0",
        ],
    },
)