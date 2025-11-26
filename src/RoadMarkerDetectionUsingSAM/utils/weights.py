from pathlib import Path

SAM_MODEL_INFO = {
    "weight": {
        "path": f"checkpoints/sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }
}


PATH_ROOT = Path(__file__).resolve().parents[1]

def _redirect_path(path: str) -> Path:
    wrapped_path = Path(path)

    if wrapped_path.is_absolute():
        return wrapped_path
    
    return (PATH_ROOT / wrapped_path).resolve()

def _download(url: str, destination: Path):
    import urllib.request
    destination.parent.mkdir(parents = True, exist_ok = True)
    print(f"downloading model's weight -> {destination}...")
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading: {percent}%", end='', flush=True)
    
    urllib.request.urlretrieve(url, str(destination), reporthook=progress_hook)

def ensure_SAM_weight(weight_path = None):
    weight_info = SAM_MODEL_INFO["weight"]
    weight_path = _redirect_path(weight_info["path"]) if weight_path is None else Path(weight_path).resolve()

    if not weight_path.exists():
        if not weight_info.get("url"):
            raise FileNotFoundError(f"missing url")
        
        _download(weight_info["url"], weight_path)

    return str(weight_path)