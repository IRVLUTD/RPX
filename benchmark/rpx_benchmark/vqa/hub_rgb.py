"""Fetch one or two members from RPX's uncompressed RGB tar shards via HTTP
ranges. Normal (single-image) samples fetch only the target image, exactly
as before. In-context (two-image) samples independently fetch the target
image AND the reference image, then reconstruct Image 1 by cropping the
fetched reference frame with the sample's own reference_crop_bbox and
verifying the result against reference_crop_sha256 -- the same deterministic
crop+PNG-encode convention data/mask_annotation/visual_grounding_gt/vqa_gt/
sos_reference.py used when the hash was originally recorded."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image

from ..exceptions import DownloadError
from .contract import VQASample, normalize_shard


def image_cache_name(sample: VQASample) -> str:
    identity = json.dumps(sample.image, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return f"rgb_{digest}{Path(sample.image['member']).suffix}"


def reference_cache_name(sample: VQASample) -> str:
    """Cache key for the reconstructed (cropped) reference image -- distinct
    from image_cache_name because the crop_bbox, not just the raw locator,
    determines the resulting bytes."""
    assert sample.reference_image is not None and sample.reference_crop_bbox is not None
    identity = json.dumps(
        {"locator": sample.reference_image, "crop_bbox": list(sample.reference_crop_bbox)},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return f"ref_{digest}.png"


class HTTPRangeReader(io.RawIOBase):
    """Small seekable reader with block caching; avoids full multi-GB shard downloads."""

    def __init__(self, url: str, block_size: int = 8 * 1024 * 1024) -> None:
        self.url = url
        self.block_size = block_size
        self.position = 0
        self.length = self._content_length()
        self._block_start = -1
        self._block = b""

    def _content_length(self) -> int:
        request = urllib.request.Request(
            self.url, headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"}
        )
        with urllib.request.urlopen(request) as response:
            content_range = response.headers.get("Content-Range")
            if not content_range:
                raise OSError("Hub endpoint does not support byte ranges")
            return int(content_range.rsplit("/", 1)[1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_CUR:
            offset += self.position
        elif whence == io.SEEK_END:
            offset += self.length
        if offset < 0:
            raise OSError("negative seek")
        self.position = offset
        return offset

    def readinto(self, buffer: bytearray) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        if size < 0:
            size = self.length - self.position
        output = bytearray()
        while size and self.position < self.length:
            block_start = (self.position // self.block_size) * self.block_size
            if block_start != self._block_start:
                block_end = min(self.length - 1, block_start + self.block_size - 1)
                request = urllib.request.Request(
                    self.url,
                    headers={
                        "Range": f"bytes={block_start}-{block_end}",
                        "Accept-Encoding": "identity",
                    },
                )
                with urllib.request.urlopen(request) as response:
                    self._block = response.read()
                self._block_start = block_start
            offset = self.position - self._block_start
            chunk = self._block[offset : offset + size]
            if not chunk:
                break
            output.extend(chunk)
            self.position += len(chunk)
            size -= len(chunk)
        return bytes(output)


def hub_url(image: VQASample | dict[str, str]) -> str:
    locator = image.image if isinstance(image, VQASample) else image
    shard = normalize_shard(str(locator["shard"]))
    return (
        f"https://huggingface.co/datasets/{locator['repo_id']}/resolve/"
        f"{locator['revision']}/{shard}"
    )


def _fetch_tar_member(locator: dict[str, str]) -> bytes:
    reader = HTTPRangeReader(hub_url(locator))
    with tarfile.open(fileobj=reader, mode="r:") as archive:
        member = next((entry for entry in archive if entry.name == locator["member"]), None)
        if member is None:
            raise FileNotFoundError(locator["member"])
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(locator["member"])
        return extracted.read()


def fetch_rgb(sample: VQASample, output_root: str | Path) -> Path:
    """Fetch, dimension-check, and cache the exact TARGET image used by a VQA
    row (Image 2 for in-context rows, the only image for normal rows)."""
    target = Path(output_root) / image_cache_name(sample)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _fetch_tar_member(sample.image)
    with Image.open(io.BytesIO(payload)) as image:
        if image.size != (sample.img_w, sample.img_h):
            raise DownloadError(
                f"{sample.sample_id}: image is {image.size}, expected {(sample.img_w, sample.img_h)}",
                hint="check that the manifest and pinned Hub revision were generated together",
            )
        image.verify()
    target.write_bytes(payload)
    return target


def fetch_reference_crop(sample: VQASample, output_root: str | Path) -> Path:
    """Fetch the RAW reference (SOS) frame, crop it with the sample's own
    reference_crop_bbox (inclusive-pixel xyxy, same convention as
    answer_bbox), and verify the PNG-encoded result against
    reference_crop_sha256. Raises DownloadError on any hash mismatch --
    never silently serves an unverified Image 1."""
    if sample.reference_image is None or sample.reference_crop_bbox is None:
        raise DownloadError(
            f"{sample.sample_id}: not an in-context sample", hint="only call this for in-context rows"
        )
    target = Path(output_root) / reference_cache_name(sample)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _fetch_tar_member(sample.reference_image)
    x0, y0, x1, y1 = sample.reference_crop_bbox
    with Image.open(io.BytesIO(payload)) as raw:
        raw.verify()
    with Image.open(io.BytesIO(payload)) as raw:
        raw = raw.convert("RGB")
        if not (0 <= x0 <= x1 < raw.width and 0 <= y0 <= y1 < raw.height):
            raise DownloadError(
                f"{sample.sample_id}: reference_crop_bbox {sample.reference_crop_bbox} "
                f"is outside the fetched reference frame {raw.size}",
                hint="check that the manifest and pinned Hub revision were generated together",
            )
        cropped = raw.crop((x0, y0, x1 + 1, y1 + 1))
        buffer = io.BytesIO()
        cropped.save(buffer, format="PNG", compress_level=6)
        crop_bytes = buffer.getvalue()
    digest = hashlib.sha256(crop_bytes).hexdigest()
    if digest != sample.reference_crop_sha256:
        raise DownloadError(
            f"{sample.sample_id}: reference crop sha256 mismatch: got {digest}, "
            f"expected {sample.reference_crop_sha256}",
            hint=(
                "the reconstructed Image 1 does not match the hash recorded at generation "
                "time -- check Pillow version compatibility with sos_reference.py's PNG "
                "encoder before assuming data corruption"
            ),
        )
    target.write_bytes(crop_bytes)
    return target


def fetch_images(sample: VQASample, output_root: str | Path) -> tuple[Path, ...]:
    """Fetch every image a model call needs, in the same [reference, target]
    order as sample.images: independently fetches and verifies each locator
    (item 3) rather than assuming one implies the other."""
    if sample.is_in_context:
        return (fetch_reference_crop(sample, output_root), fetch_rgb(sample, output_root))
    return (fetch_rgb(sample, output_root),)
