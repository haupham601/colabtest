from setuptools import setup, find_packages

setup(
    name="ai_motion_transfer",
    version="1.0.0",
    description="Pose-guided AI Motion Transfer with Wan2.1 and Consistency Distillation",
    author="Antigravity Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "diffusers>=0.30.0",
        "accelerate>=0.30.0",
        "peft>=0.10.0",
        "omegaconf>=2.3.0",
        "einops>=0.7.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "gradio>=4.20.0",
    ],
)
