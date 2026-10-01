import shutil
from pathlib import Path

import kagglehub

DEST = Path("/home/demmorou/www/projetos/machine-learning/retina-seg/data/raw/idrid")

# Download latest version (kagglehub caches it under ~/.cache/kagglehub)
cached_path = Path(kagglehub.dataset_download("aaryapatel98/indian-diabetic-retinopathy-image-dataset"))

DEST.mkdir(parents=True, exist_ok=True)
for item in cached_path.iterdir():
    target = DEST / item.name
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    shutil.move(str(item), str(target))

print("Dataset copiado para:", DEST)