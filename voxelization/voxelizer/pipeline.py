# mof_voxelizer/pipeline.py
import logging
import platform
import pymatgen
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from pymatgen.core import Structure

from .io_utils import find_cif_files, save_results
from .core import voxelize_structure
from .utils import setup_logger

def process_folder(args):
    setup_logger(args.verbose)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cif_dir = Path(args.cif_dir)

    cifs = find_cif_files(cif_dir)
    if not cifs:
        logging.warning("No .cif files found in %s", cif_dir)
        return

    stats = {"processed": 0, "skipped": 0, "failed": 0}

    for cif in tqdm(cifs, desc="Voxelizing"):
        stem = cif.stem
        if not args.overwrite and (out_dir / f"{stem}_vox.npz").exists():
            stats["skipped"] += 1
            continue

        try:
            # Load
            s = Structure.from_file(str(cif))
            
            # Process
            elem_channels = [e.strip() for e in args.elem_channels.split(",") if e.strip()]
            vox, channels, box_size = voxelize_structure(
                s, grid=args.grid, Lmin=args.Lmin, 
                elem_channels=elem_channels,
                default_sigma_vox=args.sigma,
                include_charge=args.include_charge,
                normalize=args.normalize,
                map_mode=args.map_mode,
                per_atom_gauss=args.per_atom_gauss
            )
            
            # Metadata
            meta = {
                "id": stem, "cif": str(cif), "channels": channels,
                "grid": args.grid, "box_size_ang": box_size,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pymatgen": pymatgen.__version__
                }
            }
            
            # Save
            save_results(out_dir, stem, vox, channels, meta, args.save_torch)
            stats["processed"] += 1
            
        except Exception as e:
            logging.exception("Failed %s: %s", stem, e)
            stats["failed"] += 1

    logging.info("Complete: %s", stats)
