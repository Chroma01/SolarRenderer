import os

# a long video sequence fragments the CUDA caching allocator; expandable segments keep training on 8 GPUs in memory
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .cli import main  # noqa: E402

raise SystemExit(main())
