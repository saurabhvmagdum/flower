# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for the SecAgg/SecAgg+ protocol."""


import hashlib
import struct
from collections.abc import Iterator

import numpy as np

from flwr.common import NDArrayInt


def share_keys_plaintext_concat(
    src_node_id: int, dst_node_id: int, b_share: bytes, sk_share: bytes
) -> bytes:
    """Combine arguments to bytes.

    Parameters
    ----------
    src_node_id : int
        the node ID of the source.
    dst_node_id : int
        the node ID of the destination.
    b_share : bytes
        the private key share of the source sent to the destination.
    sk_share : bytes
        the secret key share of the source sent to the destination.

    Returns
    -------
    bytes
        The combined bytes of all the arguments.
    """
    return b"".join(
        [
            int.to_bytes(src_node_id, 8, "little", signed=False),
            int.to_bytes(dst_node_id, 8, "little", signed=False),
            int.to_bytes(len(b_share), 4, "little"),
            b_share,
            sk_share,
        ]
    )


def share_keys_plaintext_separate(plaintext: bytes) -> tuple[int, int, bytes, bytes]:
    """Retrieve arguments from bytes.

    Parameters
    ----------
    plaintext : bytes
        the bytes containing 4 arguments.

    Returns
    -------
    src_node_id : int
        the node ID of the source.
    dst_node_id : int
        the node ID of the destination.
    b_share : bytes
        the private key share of the source sent to the destination.
    sk_share : bytes
        the secret key share of the source sent to the destination.
    """
    src, dst, mark = (
        int.from_bytes(plaintext[:8], "little", signed=False),
        int.from_bytes(plaintext[8:16], "little", signed=False),
        int.from_bytes(plaintext[16:20], "little"),
    )
    ret = (src, dst, plaintext[20 : 20 + mark], plaintext[20 + mark :])
    return ret


def _prf_stream(seed: bytes) -> Iterator[int]:
    """Deterministic byte stream from seed using SHA-256 counter mode."""
    counter = 0
    while True:
        # Pack counter as 8-byte little-endian unsigned integer
        h = hashlib.sha256(seed + struct.pack("<Q", counter))
        yield from h.digest()
        counter += 1


def pseudo_rand_gen(
    seed: bytes, num_range: int, dimensions_list: list[tuple[int, ...]]
) -> list[NDArrayInt]:
    """Seeded pseudo-random number generator for noise generation.
    
    Uses SHA-256 in counter mode to generate a cryptographically secure, 
    deterministic byte stream from the seed, preserving full entropy.
    Assumes `num_range` is a power of two.
    """
    if (num_range & (num_range - 1)) != 0 or num_range <= 0:
        raise ValueError("num_range must be a power of two.")

    stream = _prf_stream(seed)
    num_bytes = (num_range.bit_length() + 6) // 8
    bitmask = num_range - 1
    
    masks = []
    for shape in dimensions_list:
        if len(shape) == 0:
            # Handle scalar case
            chunk = 0
            for _ in range(num_bytes):
                chunk = (chunk << 8) | next(stream)
            val = chunk & bitmask
            masks.append(np.array(val, dtype=np.int64))
        else:
            total_elements = int(np.prod(shape))
            vals = []
            for _ in range(total_elements):
                chunk = 0
                for _ in range(num_bytes):
                    chunk = (chunk << 8) | next(stream)
                vals.append(chunk & bitmask)
            masks.append(np.array(vals, dtype=np.int64).reshape(shape))
    return masks
