import os
import torch

def load_cf_model(model_name, ModelClass=None, device="cpu"):
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    load_path = os.path.join(ROOT_DIR, "src", "saved_models", f"{model_name}.pt")

    ckpt = torch.load(load_path, map_location=device)

    # ===== embeddings (luôn có) =====
    user_emb = ckpt["user_emb"].to(device)
    item_emb = ckpt["item_emb"].to(device)

    # ===== detect metadata =====
    if "extra" in ckpt:
        extra = ckpt["extra"]
    else:
        # format cũ (NGCF / LightGCN lưu tay)
        extra = {
            "num_users": ckpt.get("num_users"),
            "num_items": ckpt.get("num_items"),
            "dim": ckpt.get("dim"),
            "layers": ckpt.get("layers"),
            "n_layers": ckpt.get("n_layers"),
        }

    model = None
    if ModelClass is not None and "model_state_dict" in ckpt:
        if ModelClass.__name__.lower() == "ngcf":
            model = ModelClass(
                extra["num_users"],
                extra["num_items"],
                dim=extra["dim"],
                layers=extra["layers"],
            ).to(device)

        elif ModelClass.__name__.lower() == "lightgcn":
            model = ModelClass(
                extra["num_users"],
                extra["num_items"],
                dim=extra["dim"],
                n_layers=extra["n_layers"],
            ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

    print(f"[OK] Loaded {model_name} from {load_path}")
    return model, user_emb, item_emb, extra
