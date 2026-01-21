import argparse
import sys
from voxelizer.constants import DEFAULT_ELEM_CHANNELS, DEFAULT_GRID_SIZE, DEFAULT_LMIN, DEFAULT_SIGMA
from voxelizer.pipeline import process_folder

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MOF Voxelizer: CIF -> 3D Grid")
    p.add_argument("--cif-dir", default="repeat_cifs", help="Input directory")
    p.add_argument("--out-dir", default="voxels_publishable", help="Output directory")
    
    # Grid Parameters
    p.add_argument("--grid", type=int, default=DEFAULT_GRID_SIZE, help="Grid dimension (G)")
    p.add_argument("--Lmin", type=float, default=DEFAULT_LMIN, help="Min cell size (Å)")
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="Gaussian blur sigma")
    p.add_argument("--map-mode", choices=["fractional", "cartesian"], default="fractional")
    
    # Feature Flags
    p.add_argument("--include-charge", action="store_true", help="Include charge channel")
    p.add_argument("--per-atom-gauss", action="store_true", help="High-fidelity per-atom gaussian")
    p.add_argument("--elem-channels", default=",".join(DEFAULT_ELEM_CHANNELS))
    p.add_argument("--normalize", default="per_channel_max", 
                   choices=["none", "per_channel_max", "global_max", "sum_normalize"])
    
    # Execution Flags
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-torch", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--test", action="store_true")

    args = p.parse_args()

    if args.test:
        from voxelizer.utils import setup_logger
        from pymatgen.core import Structure, Lattice
        from voxelizer.core import voxelize_structure
        setup_logger(True)
        s = Structure(Lattice.cubic(10), ["C"], [[0.5, 0.5, 0.5]])
        vox, ch, _ = voxelize_structure(s, grid=32, Lmin=10.0)
        print(f"Test Successful. Vox shape: {vox.shape}, Channels: {ch}")
        sys.exit(0)

    process_folder(args)
