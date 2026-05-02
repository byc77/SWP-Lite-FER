# tools/confusion_to_report.py
import csv, sys, numpy as np
assert len(sys.argv)>=3, "Usage: python confusion_to_report.py <confusion_csv> <out_csv>"
cm = np.loadtxt(sys.argv[1], delimiter=',', dtype=int)
per_class = (cm.diagonal() / cm.sum(axis=1).clip(min=1)).tolist()
bacc = float(np.mean(per_class)); overall = float(cm.diagonal().sum()/cm.sum())
with open(sys.argv[2], 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['class_id','per_class_acc'])
    for i,a in enumerate(per_class): w.writerow([i, f'{a:.4f}'])
    w.writerow([]); w.writerow(['overall_acc', f'{overall:.4f}'])
    w.writerow(['balanced_acc', f'{bacc:.4f}'])
print(f'overall_acc={overall:.4f}, balanced_acc={bacc:.4f}')



