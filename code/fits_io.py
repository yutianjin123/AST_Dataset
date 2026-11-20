"""Minimal FITS image reader (numpy only).

The example cutouts in this repository are single-HDU, 101x101 images with
BITPIX = -32 (real KiDS cutouts) or -64 (simulated lenses). A dependency-free
reader keeps the example inference script runnable with just torch + numpy.
If astropy is installed, it is used instead.
"""

from pathlib import Path

import numpy as np

_DTYPE = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}


def read_fits_image(path):
    """Return the primary HDU data as a 2-D float32 numpy array."""
    try:
        from astropy.io import fits  # optional, preferred when available

        with fits.open(path) as hdul:
            return np.asarray(hdul[0].data, dtype=np.float32)
    except ImportError:
        pass

    raw = Path(path).read_bytes()
    bitpix = naxis1 = naxis2 = None
    bscale, bzero = 1.0, 0.0
    offset = 0
    found_end = False

    while offset < len(raw):
        block = raw[offset:offset + 2880]
        for i in range(0, len(block), 80):
            card = block[i:i + 80].decode("ascii", "replace").strip()
            if "=" in card:
                key, value = card.split("=", 1)
                value = value.split("/")[0].strip()
                key = key.strip()
                if key == "BITPIX":
                    bitpix = int(value)
                elif key == "NAXIS1":
                    naxis1 = int(value)
                elif key == "NAXIS2":
                    naxis2 = int(value)
                elif key == "BSCALE":
                    bscale = float(value)
                elif key == "BZERO":
                    bzero = float(value)
            if card.startswith("END"):
                found_end = True
                break
        offset += 2880
        if found_end:
            break

    if not found_end or bitpix not in _DTYPE or not naxis1 or not naxis2:
        raise ValueError(f"Unsupported or malformed FITS file: {path}")

    count = naxis1 * naxis2
    arr = np.frombuffer(raw, dtype=_DTYPE[bitpix], count=count, offset=offset)
    arr = arr.reshape(naxis2, naxis1).astype(np.float32)
    return arr * np.float32(bscale) + np.float32(bzero)
