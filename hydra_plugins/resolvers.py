from omegaconf import OmegaConf


def get_feature(target: str) -> int:
    """Resolve feature dimensions from backbone name patterns"""
    model_name = target.lower()

    if "vits" in model_name or "vit_small" in model_name:
        return 384
    if "vitb" in model_name or "vit_base" in model_name:
        return 768
    if "vitl" in model_name or "vit_large" in model_name:
        return 1024
    if "stable-diffusion-2-1" in model_name:
        return 320
    if model_name == "efficientnet_b4":
        return 128
    if model_name == "maskclip":
        return 512
    if model_name == "radio_v2.5-h":
        return 1280
    if model_name == "radio_v2.5-l":
        return 1024
    if model_name == "radio_v2.5-b":
        return 768

    raise ValueError(f"Unsupported backbone: {model_name}")


OmegaConf.register_new_resolver("get_feature", lambda name: get_feature(name))
