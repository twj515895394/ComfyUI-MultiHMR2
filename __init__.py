"""ComfyUI-MultiHMR2 custom nodes."""

import os

# This package is loaded before nodes from later custom-node directories.  Lock
# PyOpenGL to Windows' native platform now, before another plugin can import it
# while PYOPENGL_PLATFORM=egl is set.  Changing the environment after OpenGL
# has been imported is too late because PyOpenGL caches the selected platform.
if os.name == "nt" and os.environ.get("PYOPENGL_PLATFORM") in {"egl", "osmesa"}:
    os.environ.pop("PYOPENGL_PLATFORM", None)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
