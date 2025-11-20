import os
import re

RESULTS_DIR = "results_p_sweep"
P_VALUES = ["0.00", "0.05", "0.10", "0.15", "0.20", "0.25"]
MODELS = ["index", "random", "mixer"]

ACC_PATTERN = re.compile(r"Epoch\s+(\d+)\s+\|\s+test_acc=([0-9.]+)")
L1_PATTERN = re.compile(r"Layer 1 mean rate:\s*([0-9.]+)")
L2_PATTERN = re.compile(r"Layer 2 mean rate:\s*([0-9.]+)")
L3_PATTERN = re.compile(r"Layer 3 mean rate:\s*([0-9.]+)")
OV_PATTERN = re.compile(r"Overall hidden mean rate:\s*([0-9.]+)")


def parse_file(path):
    with open(path, "r") as f:
        txt = f.read()

    acc_matches = ACC_PATTERN.findall(txt)
    if not acc_matches:
        return None
    best_acc = max(float(m[1]) for m in acc_matches) * 100.0

    l1_match = L1_PATTERN.search(txt)
    l2_match = L2_PATTERN.search(txt)
    l3_match = L3_PATTERN.search(txt)
    ov_match = OV_PATTERN.search(txt)
    if not (l1_match and l2_match and l3_match and ov_match):
        return None

    l1 = float(l1_match.group(1)) * 100.0
    l2 = float(l2_match.group(1)) * 100.0
    l3 = float(l3_match.group(1)) * 100.0
    ov = float(ov_match.group(1)) * 100.0

    return best_acc, ov, l1, l2, l3


def collect_sparse_results():
    results = {}
    for model in MODELS:
        for p in P_VALUES:
            path = os.path.join(RESULTS_DIR, f"{model}_p{p}.txt")
            if os.path.exists(path):
                parsed = parse_file(path)
                if parsed is not None:
                    results[(model, p)] = parsed
    return results


def collect_dense_result():
    path = os.path.join(RESULTS_DIR, "dense.txt")
    if not os.path.exists(path):
        return None
    return parse_file(path)


def print_accuracy_table(results):
    print("\n=== Accuracy table (best test_acc per run) ===\n")
    print(f"{'p\'':>5} {'Index':>10} {'Random':>10} {'Mixer':>10}")
    print("-" * 40)
    for p in P_VALUES:
        row = f"{p:>5}"
        for model in ["index", "random", "mixer"]:
            key = (model, p)
            row += f"{results[key][0]:10.2f}" if key in results else f"{'-':>10}"
        print(row)


def print_dense_accuracy(dense_result):
    if dense_result is None:
        return
    print("\n=== Dense baseline (p' = 1.00) ===\n")
    print(f"{'p\'':>5} {'Dense':>10}")
    print("-" * 20)
    print(f"{'1.00':>5} {dense_result[0]:10.2f}")


def print_firing_tables_per_strategy(results):
    for model in ["index", "random", "mixer"]:
        print(f"\n=== Firing rates for {model.capitalize()} (percent) ===\n")
        print(f"{'p\'':>5} {'LIF1':>10} {'LIF2':>10} {'LIF3':>10} {'Overall':>10}")
        print("-" * 50)
        for p in P_VALUES:
            key = (model, p)
            if key in results:
                _, ov, l1, l2, l3 = results[key]
                print(f"{p:>5} {l1:10.2f} {l2:10.2f} {l3:10.2f} {ov:10.2f}")


def print_dense_firing_table(dense_result):
    if dense_result is None:
        return
    _, ov, l1, l2, l3 = dense_result
    print("\n=== Firing rates for Dense baseline (percent) ===\n")
    print(f"{'p\'':>5} {'LIF1':>10} {'LIF2':>10} {'LIF3':>10} {'Overall':>10}")
    print("-" * 50)
    print(f"{'1.00':>5} {l1:10.2f} {l2:10.2f} {l3:10.2f} {ov:10.2f}")


if __name__ == "__main__":
    sparse_results = collect_sparse_results()
    dense_result = collect_dense_result()
    print_accuracy_table(sparse_results)
    print_dense_accuracy(dense_result)
    print_firing_tables_per_strategy(sparse_results)
    print_dense_firing_table(dense_result)
