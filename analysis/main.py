import argparse
from mof_analysis.utils import setup_logger
from mof_analysis.workflows import run_prediction_analysis, run_structural_analysis

def main():
    p = argparse.ArgumentParser(description="Unified MOF Prediction & Structural Analysis")
    sub = p.add_subparsers(dest="command", required=True)

    
    p_preds = sub.add_parser("analyze-preds", help="Compute regression metrics, thresholds, binning")
    p_preds.add_argument("--csv", required=True)
    p_preds.add_argument("--out-dir", default="analysis_preds")
    p_preds.add_argument("--boot-iters", type=int, default=1000)

    
    p_struct = sub.add_parser("analyze-struct", help="Compare Best vs Worst Voxel Features")
    p_struct.add_argument("--csv", required=True)
    p_struct.add_argument("--cif-root", required=True)
    p_struct.add_argument("--voxel-script", required=True)
    p_struct.add_argument("--out-dir", default="analysis_struct")
    p_struct.add_argument("--top-k", type=int, default=100)
    p_struct.add_argument("--grid", type=int, default=64)

    args = p.parse_args()
    setup_logger(True)

    if args.command == "analyze-preds":
        run_prediction_analysis(args.csv, args.out_dir, args.boot_iters)
    elif args.command == "analyze-struct":
        run_structural_analysis(args.csv, args.cif_root, args.voxel_script, args.out_dir, args.top_k, args.grid)

if __name__ == "__main__":
    main()
