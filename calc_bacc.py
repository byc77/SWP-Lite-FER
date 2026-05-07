# calc_bacc.py
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confmat", type=str, required=True, help="Path to confusion matrix CSV")
    args = parser.parse_args()

    C = np.loadtxt(args.confmat, delimiter=",", dtype=int)

    total = C.sum()
    correct = np.trace(C)

    overall_acc = correct / total

    row_sum = C.sum(axis=1)
    per_class_acc = np.diag(C) / np.maximum(row_sum, 1)

    bacc = per_class_acc.mean()

    print("Confusion matrix:")
    print(C)
    print()

    print(f"overall_acc = {overall_acc:.4f} ({overall_acc * 100:.2f}%)")
    print(f"bACC        = {bacc:.4f} ({bacc * 100:.2f}%)")
    print()

    print("Per-class accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"class{i}: {acc:.4f} ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()