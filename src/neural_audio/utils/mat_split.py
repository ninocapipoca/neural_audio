"""Split oversized MATLAB v7.3 ``.mat`` references into GitHub-sized parts.

The ``aud2cor`` unit tests compare against MATLAB-generated cortical representations
stored as v7.3 (HDF5) ``.mat`` files. Each of those files is several hundred megabytes,
well over GitHub's 100 MB per-file limit, so they cannot be committed as they are.

This module splits such a file into a folder of smaller HDF5 parts and reassembles them
on demand:

* :func:`split_mat_file` slices the dataset along one axis and writes each slab to its
  own file, alongside a ``manifest.json`` describing how to put them back together.
* :func:`split_mat_directory` does that for every oversized ``.mat`` under a directory.
* :func:`load_split_mat` rebuilds the full array from a split folder.
* :func:`load_cortical_reference` is what the tests call: it takes the canonical ``.mat``
  path and reads the form selected by its ``source`` argument.

All four take an optional ``progress`` callable that is handed one human-readable line per
part, so a long split or reassembly can report where it has got to.

The parts are written with a native complex dtype (``complex64`` by default, which halves
the committed payload and stays far inside the tests' ``atol=1e-2``), so they are plain
HDF5 rather than MATLAB-readable ``.mat`` files. They carry the ``.h5`` extension to make
that explicit; only the helpers in this module are meant to read them.

Run the one-time conversion with::

    python -m neural_audio.utils.mat_split unit_tests/matlab_outputs/aud2cor
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np

__all__ = [
    'MAX_PART_BYTES',
    'SPLIT_THRESHOLD_BYTES',
    'SOURCE_CHOICES',
    'split_mat_file',
    'split_mat_directory',
    'load_split_mat',
    'load_cortical_reference',
    'reference_exists',
    'split_dir_for',
]

#: Target ceiling for a single part. Kept under GitHub's hard 100 MB limit so that the
#: gzip-compressed result still has room to spare.
MAX_PART_BYTES = 90 * 1024 ** 2

#: GitHub's hard per-file limit. Files above this get split; parts above it are an error.
SPLIT_THRESHOLD_BYTES = 100 * 1024 ** 2

#: Accepted values for the ``source`` argument of :func:`load_cortical_reference` and
#: :func:`reference_exists`. ``'split'`` reads only the split parts, ``'mat'`` only the
#: original ``.mat``, ``'auto'`` prefers the parts and falls back to the ``.mat``.
SOURCE_CHOICES = ('split', 'mat', 'auto')

MANIFEST_NAME = 'manifest.json'
_PART_TEMPLATE = 'part_{:03d}.h5'


def _noop(message: str) -> None:
    """Default ``progress`` sink: report nothing."""


def _flushing_print(message: str) -> None:
    """``progress`` sink that prints immediately, so long jobs report as they go."""
    print(message, flush=True)


def _check_source(source: str) -> str:
    if source not in SOURCE_CHOICES:
        raise ValueError(
            f"source must be one of {SOURCE_CHOICES}, got {source!r}"
        )
    return source


def split_dir_for(mat_path) -> Path:
    """Folder that holds (or would hold) the split parts of ``mat_path``.

    ``.../cr_halfres_geese.mat`` maps to ``.../cr_halfres_geese``.
    """
    mat_path = Path(mat_path)
    return mat_path.parent / mat_path.stem


def _sha256(path: Path, block_size: int = 8 * 1024 ** 2) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(block_size), b''):
            digest.update(block)
    return digest.hexdigest()


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


def _slab_width(shape, axis: int, itemsize: int, max_bytes: int,
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


def split_mat_file(mat_path, *, dataset: str = 'cr', axis: int = 0,
                   max_bytes: int = MAX_PART_BYTES, dtype=np.complex64,
                   overwrite: bool = False, checksum: bool = True,
                   progress: Callable[[str], None] | None = None) -> Path:
    """Split one ``.mat`` file into a folder of sub-100 MB HDF5 parts.

    Parameters
    ----------
    mat_path : path-like
        The v7.3 ``.mat`` file to split.
    dataset : str
        Name of the dataset inside the file. The aud2cor references store one array
        called ``cr``.
    axis : int
        Axis to slice along, in HDF5 order. Axis 0 of the aud2cor references is the
        128 frequency channels.
    max_bytes : int
        Target ceiling for each part, uncompressed.
    dtype : numpy dtype
        Dtype the parts are stored in. ``complex64`` halves the size relative to MATLAB's
        ``complex128`` and is far more precise than the tests' tolerance requires.
    overwrite : bool
        Replace an existing split folder instead of raising.
    checksum : bool
        Record sha256 digests of the source and each part in the manifest.
    progress : callable, optional
        Called with one line per part as it is written, and once more when the manifest
        lands. Pass ``print`` to watch a long split.

    Returns
    -------
    pathlib.Path
        The folder containing the parts and their manifest.
    """
    mat_path = Path(mat_path)
    if not mat_path.is_file():
        raise FileNotFoundError(f"no such .mat file: {mat_path}")

    progress = progress or _noop
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
        progress(
            f"{mat_path.name}: {shape} {source_dtype} -> {n_parts} x {dtype.name} parts "
            f"of {width} along axis {axis}"
        )

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
            if checksum:
                parts[-1]['sha256'] = _sha256(part_path)

            progress(
                f"  wrote {part_name}  [{index + 1}/{n_parts}]  {start}:{stop}  "
                f"{parts[-1]['bytes'] / 1024 ** 2:.1f} MB"
            )

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
    if checksum:
        manifest['source_sha256'] = _sha256(mat_path)

    with open(out_dir / MANIFEST_NAME, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)

    total = sum(p['bytes'] for p in parts) / 1024 ** 2
    largest = max(p['bytes'] for p in parts) / 1024 ** 2
    progress(
        f"  {out_dir.name}/: {len(parts)} parts, {total:.1f} MB total, "
        f"largest {largest:.1f} MB"
    )

    return out_dir


def split_mat_directory(root, *, threshold_bytes: int = SPLIT_THRESHOLD_BYTES,
                        verbose: bool = True, **kwargs) -> list[Path]:
    """Split every ``.mat`` under ``root`` that is larger than ``threshold_bytes``.

    Recurses into subdirectories, so ``matlab_outputs/aud2cor`` covers ``synthetic/`` too.
    Files that already have a split folder are skipped unless ``overwrite=True`` is passed
    through to :func:`split_mat_file`.

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
                print(f"skip   {mat_path.name}: {size / 1024 ** 2:.0f} MB, under threshold")
            continue

        if split_dir_for(mat_path).exists() and not kwargs.get('overwrite', False):
            if verbose:
                print(f"skip   {mat_path.name}: already split")
            continue

        if verbose:
            print(f"split  {mat_path.name}: {size / 1024 ** 2:.0f} MB ...", flush=True)
        kwargs.setdefault('progress', _flushing_print if verbose else None)
        out_dir = split_mat_file(mat_path, **kwargs)
        written.append(out_dir)

    return written


def load_split_mat(part_dir, *, dataset: str | None = None, dtype=None,
                   progress: Callable[[str], None] | None = None) -> np.ndarray:
    """Rebuild the full array from a folder written by :func:`split_mat_file`.

    The array comes back in HDF5 axis order, i.e. exactly what reading the original
    ``.mat`` with ``h5py`` would give. Callers that need MATLAB's axis order still have to
    reverse the axes themselves.

    ``dtype`` overrides the stored dtype; leaving it as ``None`` returns the parts as
    written (``complex64``), which keeps peak memory down. ``progress`` is called with one
    line per part as it is read.
    """
    part_dir = Path(part_dir)
    progress = progress or _noop
    manifest_path = part_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in {part_dir}")

    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)

    dataset = dataset or manifest['dataset']
    shape = tuple(manifest['shape'])
    axis = manifest['split_axis']
    out = np.empty(shape, dtype=np.dtype(dtype or manifest['dtype']))

    n_parts = len(manifest['parts'])
    progress(
        f"{part_dir.name}/: reassembling {n_parts} parts -> {shape} {out.dtype.name}"
    )

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

        progress(
            f"  read {entry['file']}  [{part_number}/{n_parts}]  "
            f"{entry['start']}:{entry['stop']}"
        )
        expected_start = entry['stop']

    if expected_start != shape[axis]:
        raise ValueError(
            f"{part_dir.name}: parts cover {expected_start} of {shape[axis]} indices "
            f"along axis {axis}"
        )

    return out


def load_cortical_reference(mat_path, *, dataset: str = 'cr', dtype=None,
                            source: str = 'split',
                            progress: Callable[[str], None] | None = None) -> np.ndarray:
    """Load a MATLAB reference in the form named by ``source``.

    ``source='split'`` (the default) reads only the split parts and raises if they are
    absent — an original ``.mat`` sitting next to them is deliberately ignored, so a run
    cannot quietly bypass the split machinery just because the big files happen to be on
    this machine. ``source='mat'`` reads only the original, and ``source='auto'`` prefers
    the parts and falls back to the ``.mat``.

    Returns the array in HDF5 axis order.
    """
    mat_path = Path(mat_path)
    _check_source(source)
    progress = progress or _noop

    part_dir = split_dir_for(mat_path)
    have_parts = (part_dir / MANIFEST_NAME).is_file()

    if source in ('split', 'auto') and have_parts:
        return load_split_mat(part_dir, dataset=dataset, dtype=dtype, progress=progress)

    if source == 'split':
        raise FileNotFoundError(
            f"no split parts at {part_dir}. Build them with "
            f"`python -m neural_audio.utils.mat_split {mat_path.parent}`, or pass "
            f"source='mat'/'auto' to read the original .mat"
        )

    if mat_path.is_file():
        progress(f"{mat_path.name}: reading original .mat")
        with h5py.File(mat_path, 'r') as source_file:
            if dataset not in source_file:
                raise KeyError(
                    f"{mat_path.name} has no dataset {dataset!r} "
                    f"(found: {sorted(source_file.keys())})"
                )
            src = source_file[dataset]
            target = np.dtype(dtype) if dtype is not None else np.dtype(np.complex128)
            out = np.empty(src.shape, dtype=target)

            # Slab-wise so MATLAB's 16-bytes-per-element compound layout is never fully
            # resident on top of the complex result.
            chunk_extent = src.chunks[0] if src.chunks else 1
            width = _slab_width(src.shape, 0, target.itemsize, MAX_PART_BYTES, chunk_extent)
            n_slabs = -(-src.shape[0] // width)
            for slab_number, start in enumerate(range(0, src.shape[0], width), start=1):
                stop = min(start + width, src.shape[0])
                out[start:stop] = _read_complex_slab(src, 0, start, stop, target)
                progress(f"  read slab [{slab_number}/{n_slabs}]  {start}:{stop}")
            return out

    if source == 'mat':
        raise FileNotFoundError(f"no original .mat at {mat_path}")

    raise FileNotFoundError(
        f"neither {mat_path} nor a split folder at {part_dir} was found"
    )


def reference_exists(mat_path, *, source: str = 'split') -> bool:
    """True when a reference is available in the form named by ``source``.

    Mirrors :func:`load_cortical_reference`, so a caller can guard with the same policy it
    will load under: under ``'split'`` an original ``.mat`` alone does not count.
    """
    mat_path = Path(mat_path)
    _check_source(source)

    have_parts = (split_dir_for(mat_path) / MANIFEST_NAME).is_file()
    if source == 'split':
        return have_parts
    if source == 'mat':
        return mat_path.is_file()
    return have_parts or mat_path.is_file()


def _main(argv=None) -> int:
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
    parser.add_argument('--no-checksum', action='store_true', help='skip sha256 digests')
    args = parser.parse_args(argv)

    written = split_mat_directory(
        args.root,
        threshold_bytes=args.threshold_bytes,
        dataset=args.dataset,
        axis=args.axis,
        max_bytes=args.max_bytes,
        dtype=np.dtype(args.dtype),
        overwrite=args.overwrite,
        checksum=not args.no_checksum,
    )
    print(f"\n{len(written)} file(s) split.")
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
