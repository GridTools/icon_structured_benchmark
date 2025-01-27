from skbuild import setup

setup(
    name="icon_structured_benchmark",
    version="0.1.1",
    description="A Python package with a CMake project",
    author="Your Name",
    packages=["icon_benchmark"],  # Replace with your package name
    package_dir={"icon_benchmark": "python"},
    cmake_install_dir="python",
    # packages=["."],  # Replace with your package name
    # cmake_install_dir="build_skbuild",  # Directory to install built files
)
