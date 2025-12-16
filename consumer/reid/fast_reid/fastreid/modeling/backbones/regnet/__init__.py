

# from .regnet import build_regnet_backbone
# from .effnet import build_effnet_backbone

def build_regnet_backbone(cfg):
    """Lazy import để tránh circular dependency"""
    from .regnet import build_regnet_backbone as _build
    return _build(cfg)

def build_effnet_backbone(cfg):
    """Lazy import để tránh circular dependency"""
    from .effnet import build_effnet_backbone as _build
    return _build(cfg)

__all__ = ['build_regnet_backbone', 'build_effnet_backbone']