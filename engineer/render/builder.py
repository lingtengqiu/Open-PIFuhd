# author: lingtengqiu
# data: 2021-01-18
# all networks are built from two part:
# one is backbone, the other is head
from engineer.registry import build_from_cfg
from .registry import NORMAL_RENDER

def build(cfg, registry, default_args=None):
    return build_from_cfg(cfg, registry, default_args)

def build_render(cfg):
    return build(cfg, NORMAL_RENDER)
