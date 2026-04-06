"""HuggingFace Spaces entry point — launches the Moleculyst HTTP server."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agent_banana.server import main

if __name__ == "__main__":
    # HF Spaces exposes port 7860 by default
    sys.argv = ["app", "--host", "0.0.0.0", "--port", "7860"]
    main()
