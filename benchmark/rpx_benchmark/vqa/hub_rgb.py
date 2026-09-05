"""Fetch one member from RPX's uncompressed RGB tar shards via HTTP ranges."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image

from ..exceptions import DownloadError
from .contract import VQASample


def image_cache_name(sample: VQASample) -> str:
    identity = json.dumps(sample.image, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return f"rgb_{digest}{Path(sample.image['member']).suffix}"


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


def hub_url(sample: VQASample) -> str:
    image = sample.image
    return (
        f"https://huggingface.co/datasets/{image['repo_id']}/resolve/"
        f"{image['revision']}/{image['shard']}"
    )


def fetch_rgb(sample: VQASample, output_root: str | Path) -> Path:
    """Fetch, dimension-check, and cache the exact image used by a VQA row."""
    target = Path(output_root) / image_cache_name(sample)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    reader = HTTPRangeReader(hub_url(sample))
    with tarfile.open(fileobj=reader, mode="r:") as archive:
        member = next(
            (entry for entry in archive if entry.name == sample.image["member"]),
            None,
        )
        if member is None:
            raise FileNotFoundError(sample.image["member"])
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(sample.image["member"])
        payload = extracted.read()
    with Image.open(io.BytesIO(payload)) as image:
        if image.size != (sample.img_w, sample.img_h):
            raise DownloadError(
                f"{sample.sample_id}: image is {image.size}, expected {(sample.img_w, sample.img_h)}",
                hint="check that the manifest and pinned Hub revision were generated together",
            )
        image.verify()
    target.write_bytes(payload)
    return target
