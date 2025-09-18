python - <<'PY'
import os
import shutil
import time
from PIL import Image
import app


def run_loader(path, loader, mode, delta):
    backup = f"{path}.bak_test"
    if os.path.exists(backup):
        os.remove(backup)
    shutil.copy2(path, backup)
    try:
        first = loader()
        second = loader()
        assert first is not second, "Loader should return a fresh Image copy each call"
        base_size = first.size
        print(f"Initial size for {path}: {base_size}")
        time.sleep(1.1)
        new_size = (base_size[0] + delta[0], base_size[1] + delta[1])
        color = (255, 0, 0, 255) if mode == "RGBA" else (255, 0, 0)
        Image.new(mode, new_size, color=color).save(path)
        updated = loader()
        print(f"Updated size for {path}: {updated.size}")
        assert updated.size == new_size, f"Expected {new_size}, got {updated.size}"
    finally:
        os.replace(backup, path)


run_loader(
    os.path.join("assets", "guides", "tshirts", "STANDARD.png"),
    lambda: app.load_guide_image("tshirts", "STANDARD"),
    "RGBA",
    (11, 7),
)

run_loader(
    os.path.join("assets", "tshirts", "WHITE.jpg"),
    lambda: app.load_shirt_image("tshirts", "WHITE"),
    "RGB",
    (13, 9),
)
print("Verification completed.")
PY
```​:codex-terminal-citation[codex-terminal-citation]{line_range_start=1 line_range_end=26 terminal_chunk_id=161c6c}​
