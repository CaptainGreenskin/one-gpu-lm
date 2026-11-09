"""Compatibility imports for the separated model adapter and local service."""
from one_gpu_lm.models.hf_runtime import Runtime
from one_gpu_lm.serving.client import MAX_REQUEST_BYTES, request
from one_gpu_lm.serving.local_backend import main


if __name__ == "__main__":
    main()
