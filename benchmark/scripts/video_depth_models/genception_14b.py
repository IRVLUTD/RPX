from .genception import GenCeptionVideoDepth


def build(device: str = "cuda", **kwargs: object) -> GenCeptionVideoDepth:
    return GenCeptionVideoDepth(device=device, variant="14b", **kwargs)
