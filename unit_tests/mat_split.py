"""Split MATLAB v7.3 ``.mat`` references into GitHub-sized parts.

The ``aud2cor`` unit tests compare against MATLAB-generated cortical representations stored
as v7.3 (HDF5) ``.mat`` files. Each of those is several hundred megabytes, well over
GitHub's 100 MB per-file limit, so they cannot be committed as they are. What is committed
instead is a folder of smaller HDF5 parts per reference, which this module reassembles on
demand.

Since the split files have been comitted to the repository, this file should only be useful
when adding test cases, or modifying existing cases such that their MATLAB version needs to be
produced from scratch.

Reading — what the tests use:

* :func:`load_cortical_reference` is what the tests call: it takes the canonical ``.mat``
  path and returns the array reassembled from the parts committed next to it.
* :func:`reference_exists` says whether those parts are there, so a case can be skipped.
* :func:`load_split_mat` rebuilds the full array from a split folder.
* :func:`split_dir_for` maps a ``.mat`` path to its split folder.

Writing — maintainer only, once per reference:

* :func:`split_mat_file` slices the dataset along one axis and writes each slab to its own
  file, alongside a ``manifest.json`` describing how to put them back together.
* :func:`split_mat_directory` does that for every large ``.mat`` under a directory.

The parts are written using ``complex64`` by default, so they are plain
HDF5 and are not designed to be MATLAB-readable; only the helpers in this module are meant to read them.

Run the one-time conversion from the repository root with::

    python -m unit_tests.mat_split unit_tests/matlab_outputs/aud2cor
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

__all__ = [
    'MAX_PART_BYTES',
    'SPLIT_THRESHOLD_BYTES',
    'split_mat_file',
    'split_mat_directory',
    'load_split_mat',
    'load_cortical_reference',
    'reference_exists',
    'split_dir_for',
]

#: Target ceiling for a single part. Kept under GitHub's hard 100 MB limit with some room to spare
MAX_PART_BYTES = 90 * 1024 ** 2

#: GitHub's hard per-file limit. Files above this get split; parts above it are an error.
SPLIT_THRESHOLD_BYTES = 100 * 1024 ** 2

MANIFEST_NAME = 'manifest.json'
_PART_TEMPLATE = 'part_{:03d}.h5'


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------

def _log(message: str) -> None:
    """Progress line on stderr, where unittest and pytest write their own output."""
    print(message, file=sys.stderr, flush=True)


def split_dir_for(mat_path: str | Path) -> Path:
    """Folder that holds (or would hold) the split parts of ``mat_path``.
    eg, ``.../cr_halfres_geese.mat`` maps to ``.../cr_halfres_geese``.
    """
    mat_path = Path(mat_path)
    return mat_path.parent / mat_path.stem


def _is_matlab_complex(dtype: np.dtype) -> bool:
    """True for MATLAB's split-complex storage, a compound ``('real', 'imag')`` dtype."""
    return dtype.names is not None and set(dtype.names) == {'real', 'imag'}


def _read_complex_slab(dataset: h5py.Dataset, axis: int, start: int, stop: int,
                       dtype: np.dtype) -> np.ndarray:
    """Read ``dataset[start:stop]`` along ``axis`` as a plain complex array.

    MATLAB stores complex data as a compound dtype with ``real``/``imag`` fields; this
    reassembles those into ``dtype`` without ever materialising a full-size intermediate.
    """
    index = [slice(None)] * dataset.ndim
    index[axis] = slice(start, stop)
    raw = dataset[tuple(index)]

    if not _is_matlab_complex(raw.dtype):
        return raw.astype(dtype, copy=False)

    slab = np.empty(raw.shape, dtype=dtype)
    slab.real = raw['real']
    slab.imag = raw['imag']
    return slab


def _slab_width(shape: tuple[int, ...], axis: int, itemsize: int, max_bytes: int,
                chunk_extent: int) -> int:
    """Number of indices along ``axis`` that fit in ``max_bytes``.

    Rounded down to a multiple of ``chunk_extent`` where possible, so that reads from the
    source line up with its HDF5 chunk boundaries and stay cheap to decompress.
    """
    bytes_per_index = itemsize
    for dim, length in enumerate(shape):
        if dim != axis:
            bytes_per_index *= length

    width = max_bytes // max(bytes_per_index, 1)
    if width < 1:
        # A single index along the axis already exceeds the budget; nothing to do but
        # emit it and let the post-write size check complain if it is genuinely too big.
        return 1

    if chunk_extent > 1 and width >= chunk_extent:
        width -= width % chunk_extent

    return int(min(width, shape[axis]))


# --------------------------------------------------------------------------------------
# Reading (for aud2cor_split_tests)
# --------------------------------------------------------------------------------------

def reference_exists(mat_path: str | Path) -> bool:
    """True when the split parts for ``mat_path`` are committed next to it.

    Mirrors the default policy of :func:`load_cortical_reference`, so a caller can guard
    with the same rule it will load under: an original ``.mat`` alone does not count.
    """
    return (split_dir_for(mat_path) / MANIFEST_NAME).is_file()


def load_split_mat(part_dir: str | Path, *, verbose: bool = False) -> np.ndarray:
    """Rebuild the full array from a folder written by :func:`split_mat_file`.

    The array comes back in HDF5 axis order, i.e. exactly what reading the original
    ``.mat`` with ``h5py`` would give, and in the dtype the parts were stored in
    (``complex64``). Callers that need MATLAB's axis order still have to reverse the axes
    themselves.
    """
    part_dir = Path(part_dir)
    manifest_path = part_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {part_dir}")

    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)

    dataset = manifest['dataset']
    shape = tuple(manifest['shape'])
    axis = manifest['split_axis']
    out = np.empty(shape, dtype=np.dtype(manifest['dtype']))

    n_parts = len(manifest['parts'])
    if verbose:
        _log(f"{part_dir.name}/: reassembling {n_parts} parts -> {shape} {out.dtype.name}")

    expected_start = 0
    for part_number, entry in enumerate(manifest['parts'], start=1):
        part_path = part_dir / entry['file']
        if not part_path.is_file():
            raise FileNotFoundError(f"manifest lists {entry['file']}, missing from {part_dir}")
        if entry['start'] != expected_start:
            raise ValueError(
                f"{part_dir.name}: parts do not tile axis {axis} — {entry['file']} starts "
                f"at {entry['start']}, expected {expected_start}"
            )

        index = [slice(None)] * out.ndim
        index[axis] = slice(entry['start'], entry['stop'])
        dest = out[tuple(index)]
        with h5py.File(part_path, 'r') as part_file:
            if dataset not in part_file:
                raise KeyError(f"{part_path.name} has no dataset {dataset!r}")
            part = part_file[dataset]
            if dest.flags['C_CONTIGUOUS'] and dest.dtype == part.dtype:
                # Straight into the destination slice, no slab-sized temporary.
                part.read_direct(dest)
            else:
                dest[...] = part[...]

        if verbose:
            _log(f"  read {entry['file']}  [{part_number}/{n_parts}]  "
                 f"{entry['start']}:{entry['stop']}")
        expected_start = entry['stop']

    if expected_start != shape[axis]:
        raise ValueError(
            f"{part_dir.name}: parts cover {expected_start} of {shape[axis]} indices "
            f"along axis {axis}"
        )

    return out


def load_cortical_reference(mat_path: str | Path, *, dataset: str = 'cr',
                            force_matlab: bool = False,
                            verbose: bool = False) -> np.ndarray:
    """Load a MATLAB cortical reference, in HDF5 axis order.

    Reads the committed split parts and raises if they are absent — an original ``.mat``
    sitting next to them is deliberately ignored, so a run cannot quietly bypass the split
    machinery just because the big files happen to be on this machine. The parts come back
    as ``complex64``, the dtype they were stored in.

    ``force_matlab=True`` reads the original ``.mat`` instead, at ``complex128``. This is a
    dev/debug escape hatch for confirming a failure is in ``aud2cor`` rather than in the
    splitting; the originals are gitignored and not part of a clone, so the test suite
    never uses it.
    """
    mat_path = Path(mat_path)

    if not force_matlab:
        part_dir = split_dir_for(mat_path)
        if not (part_dir / MANIFEST_NAME).is_file():
            raise FileNotFoundError(
                f"no split parts at {part_dir}. Build them from the repository root with "
                f"`python -m unit_tests.mat_split {mat_path.parent}`, or pass "
                f"force_matlab=True to read the original .mat"
            )
        return load_split_mat(part_dir, verbose=verbose)

    if not mat_path.is_file():
        raise FileNotFoundError(f"no original .mat at {mat_path}")

    if verbose:
        _log(f"{mat_path.name}: reading original .mat")

    with h5py.File(mat_path, 'r') as source_file:
        if dataset not in source_file:
            raise KeyError(
                f"{mat_path.name} has no dataset {dataset!r} "
                f"(found: {sorted(source_file.keys())})"
            )
        src = source_file[dataset]
        target = np.dtype(np.complex128)
        out = np.empty(src.shape, dtype=target)

        # Slab-wise so MATLAB's 16-bytes-per-element compound layout is never fully
        # resident on top of the complex result.
        chunk_extent = src.chunks[0] if src.chunks else 1
        width = _slab_width(src.shape, 0, target.itemsize, MAX_PART_BYTES, chunk_extent)
        n_slabs = -(-src.shape[0] // width)
        for slab_number, start in enumerate(range(0, src.shape[0], width), start=1):
            stop = min(start + width, src.shape[0])
            out[start:stop] = _read_complex_slab(src, 0, start, stop, target)
            if verbose:
                _log(f"  read slab [{slab_number}/{n_slabs}]  {start}:{stop}")

    return out


# --------------------------------------------------------------------------------------
# Writing — maintainer only, once per reference
# --------------------------------------------------------------------------------------

def split_mat_file(mat_path: str | Path, *, dataset: str = 'cr', axis: int = 0,
                   max_bytes: int = MAX_PART_BYTES, dtype: np.dtype | type = np.complex64,
                   overwrite: bool = False, verbose: bool = False) -> Path:
    """Split one ``.mat`` file into a folder of sub-100 MB HDF5 parts.

    Parameters
    ----------
    mat_path : str or pathlib.Path
        The v7.3 ``.mat`` file to split.
    dataset : str
        Name of the dataset inside the file. The aud2cor references store one array
        called ``cr``.
    axis : int
        Axis to slice along, in HDF5 order. Axis 0 of the aud2cor references is the
        128 frequency channels.
    max_bytes : int
        Target ceiling for each part, uncompressed.
    dtype : numpy.dtype
        Dtype the parts are stored in. ``complex64`` halves the size relative to MATLAB's
        ``complex128`` and is far more precise than the tests' tolerance requires.
    overwrite : bool
        Replace an existing split folder instead of raising.
    verbose : bool
        Report one line per part on stderr as it is written, and once more when the
        manifest lands.

    Returns
    -------
    pathlib.Path
        The folder containing the parts and their manifest.
    """
    mat_path = Path(mat_path)
    if not mat_path.is_file():
        raise FileNotFoundError(f"no such .mat file: {mat_path}")

    out_dir = split_dir_for(mat_path)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"{out_dir} already exists; pass overwrite=True to rebuild it"
            )
        for stale in sorted(out_dir.glob('*')):
            stale.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = np.dtype(dtype)

    with h5py.File(mat_path, 'r') as source:
        if dataset not in source:
            raise KeyError(
                f"{mat_path.name} has no dataset {dataset!r} "
                f"(found: {sorted(source.keys())})"
            )
        src = source[dataset]
        shape = tuple(src.shape)
        if not -src.ndim <= axis < src.ndim:
            raise ValueError(f"axis {axis} out of range for shape {shape}")
        axis = axis % src.ndim

        src_chunks = src.chunks
        chunk_extent = src_chunks[axis] if src_chunks else 1
        width = _slab_width(shape, axis, dtype.itemsize, max_bytes, chunk_extent)

        # Part chunking mirrors the source, clamped so no chunk dim exceeds its part.
        part_shape = list(shape)
        part_shape[axis] = width
        if src_chunks:
            chunks = tuple(min(c, s) for c, s in zip(src_chunks, part_shape))
        else:
            chunks = None

        source_dtype = 'complex128' if _is_matlab_complex(src.dtype) else str(src.dtype)

        n_parts = -(-shape[axis] // width)   # ceil
        if verbose:
            _log(f"{mat_path.name}: {shape} {source_dtype} -> {n_parts} x {dtype.name} "
                 f"parts of {width} along axis {axis}")

        parts = []
        for index, start in enumerate(range(0, shape[axis], width)):
            stop = min(start + width, shape[axis])
            slab = _read_complex_slab(src, axis, start, stop, dtype)

            part_name = _PART_TEMPLATE.format(index)
            part_path = out_dir / part_name
            slab_chunks = (
                tuple(min(c, s) for c, s in zip(chunks, slab.shape)) if chunks else None
            )
            with h5py.File(part_path, 'w') as part_file:
                part_file.create_dataset(
                    dataset, data=slab, chunks=slab_chunks, compression='gzip',
                )
            del slab

            parts.append({
                'file': part_name,
                'start': int(start),
                'stop': int(stop),
                'bytes': part_path.stat().st_size,
            })

            if verbose:
                _log(f"  wrote {part_name}  [{index + 1}/{n_parts}]  {start}:{stop}  "
                     f"{parts[-1]['bytes'] / 1024 ** 2:.1f} MB")

    oversized = [p for p in parts if p['bytes'] > SPLIT_THRESHOLD_BYTES]
    if oversized:
        names = ', '.join(p['file'] for p in oversized)
        raise RuntimeError(
            f"{mat_path.name}: parts exceed the {SPLIT_THRESHOLD_BYTES} byte limit "
            f"({names}); lower max_bytes and split again"
        )

    manifest = {
        'source': mat_path.name,
        'dataset': dataset,
        'shape': list(shape),
        'split_axis': axis,
        'dtype': dtype.name,
        'source_dtype': source_dtype,
        'parts': parts,
    }

    with open(out_dir / MANIFEST_NAME, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)

    if verbose:
        total = sum(p['bytes'] for p in parts) / 1024 ** 2
        largest = max(p['bytes'] for p in parts) / 1024 ** 2
        _log(f"  {out_dir.name}/: {len(parts)} parts, {total:.1f} MB total, "
             f"largest {largest:.1f} MB")

    return out_dir


def split_mat_directory(root: str | Path, *, threshold_bytes: int = SPLIT_THRESHOLD_BYTES,
                        dataset: str = 'cr', axis: int = 0,
                        max_bytes: int = MAX_PART_BYTES,
                        dtype: np.dtype | type = np.complex64,
                        overwrite: bool = False, verbose: bool = False) -> list[Path]:
    """Split every ``.mat`` under ``root`` that is larger than ``threshold_bytes``.

    Recurses into subdirectories, so ``matlab_outputs/aud2cor`` covers ``synthetic/`` too.
    Files that already have a split folder are skipped unless ``overwrite=True``. Every
    other argument is passed straight through to :func:`split_mat_file`.

    Returns the list of folders written.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"no such directory: {root}")

    written = []
    for mat_path in sorted(root.rglob('*.mat')):
        size = mat_path.stat().st_size
        if size <= threshold_bytes:
            if verbose:
                _log(f"skip   {mat_path.name}: {size / 1024 ** 2:.0f} MB, under threshold")
            continue

        if split_dir_for(mat_path).exists() and not overwrite:
            if verbose:
                _log(f"skip   {mat_path.name}: already split")
            continue

        if verbose:
            _log(f"split  {mat_path.name}: {size / 1024 ** 2:.0f} MB ...")

        written.append(split_mat_file(
            mat_path, dataset=dataset, axis=axis, max_bytes=max_bytes, dtype=dtype,
            overwrite=overwrite, verbose=verbose,
        ))

    return written


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Split oversized MATLAB v7.3 .mat files into GitHub-sized parts.',
    )
    parser.add_argument(
        'root', type=Path,
        help='directory to scan recursively for .mat files (e.g. unit_tests/matlab_outputs/aud2cor)',
    )
    parser.add_argument('--dataset', default='cr', help='dataset name inside each file')
    parser.add_argument('--axis', type=int, default=0, help='axis to split along, HDF5 order')
    parser.add_argument(
        '--max-bytes', type=int, default=MAX_PART_BYTES,
        help=f'target ceiling per part in bytes (default {MAX_PART_BYTES})',
    )
    parser.add_argument(
        '--threshold-bytes', type=int, default=SPLIT_THRESHOLD_BYTES,
        help='only split files larger than this',
    )
    parser.add_argument('--dtype', default='complex64', help='dtype to store parts in')
    parser.add_argument('--overwrite', action='store_true', help='rebuild existing split folders')
    args = parser.parse_args(argv)

    written = split_mat_directory(
        args.root,
        threshold_bytes=args.threshold_bytes,
        dataset=args.dataset,
        axis=args.axis,
        max_bytes=args.max_bytes,
        dtype=np.dtype(args.dtype),
        overwrite=args.overwrite,
        verbose=True,
    )
    print(f"\n{len(written)} file(s) split.")
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
