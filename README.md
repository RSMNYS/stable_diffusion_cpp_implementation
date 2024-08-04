# Stable Diffusion Image Generator

This repository contains the code to build and run an image generator using TensorFlow Lite and Stable Diffusion. The following instructions will guide you through setting up the environment, installing dependencies, compiling the code with Bazel, and executing the binary.

## Prerequisites

Ensure you have the following installed:
- Python 3.x
- Bazel
- A compatible C++ compiler

## Setup Instructions

1. **Create and Activate a Python Virtual Environment**

    Create a virtual environment to isolate your Python dependencies:

    ```sh
    python -m venv .env
    ```

    Activate the virtual environment:

    - On macOS and Linux:
      ```sh
      source .env/bin/activate
      ```

    - On Windows:
      ```sh
      .env\Scripts\activate
      ```

2. **Install Python Dependencies**

    Install the required Python dependencies from the `requirements.txt` file:

    ```sh
    pip install -r requirements.txt
    ```

3. **Build the Project with Bazel**

    Compile the project using Bazel. Ensure you add the `--experimental_repo_remote_exec` flag:

    ```sh
    bazel build -c opt image_generator --experimental_repo_remote_exec
    ```

4. **Run the Image Generator**

    Execute the compiled binary to run the image generator:

    ```sh
    bazel-bin/image_generator
    ```

5. **Convert raw image to png**

    ```sh
    magick -depth 8 -size 512x512+0 rgb:generated_image.raw decoded.png
    ```

## Troubleshooting

- If you encounter issues related to missing dependencies, ensure that you have installed all required packages and that your virtual environment is activated.
- Ensure that Bazel and your C++ compiler are properly installed and configured in your system's PATH.