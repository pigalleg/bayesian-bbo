# Python Environments and Notebook Kernels

- When a notebook needs a kernel or the requested kernel cannot be found, search the repository for an existing `.venv` and use its Python interpreter/kernel when suitable.
- Never create a virtual environment yourself, including with `python -m venv`, `uv venv`, or equivalent commands. If no suitable environment exists, ask the user how they want to proceed.
