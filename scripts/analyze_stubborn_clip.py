"""
Analyze the sweep CSV to characterize the stubborn clip's residual across all 180 combos.
"""
import pandas as pd

df = pd.read_csv("data/processed/tier1_tuning/sweep_results.csv")

stub = "202101141947512003470I1.avi"
stub_col = f"smoothed__{stub}"

no_spurious = df[df["spurious_new"] == 0.0].copy()

print("=" * 72)
print("Stubborn-clip analysis: 202101141947512003470I1.avi")
print("Raw baseline rate: 216.00 switches/min")
print("=" * 72)

if stub_col in df.columns:
    print(f"\nDistribution of smoothed rates for this clip (spurious=0 only):")
    print(no_spurious[stub_col].value_counts().sort_index().to_string())

    best_for_stub = no_spurious.sort_values(stub_col).head(10)
    print(f"\nTop-10 combos minimising this clip's residual (spurious=0):")
    cols = ["alpha", "switch_threshold", "min_dwell_frames", "dwell_ms",
            stub_col, "total_residual", "spurious_new"]
    print(best_for_stub[cols].to_string(index=False))

    minimum = no_spurious[stub_col].min()
    print(f"\nMinimum achievable smoothed rate for this clip (spurious=0): {minimum:.2f}/min")
    print("(If minimum > 0, this clip cannot be fully suppressed within the current "
          "parameter grid without introducing spurious switches.)")
else:
    print(f"\nColumn '{stub_col}' not found in CSV. Available columns:")
    print([c for c in df.columns if "smoothed__" in c])

print("\n" + "=" * 72)
print("Global sweep statistics (spurious=0):")
print(f"  min total_residual : {no_spurious['total_residual'].min():.2f}/min")
print(f"  min max_residual   : {no_spurious['max_residual'].min():.2f}/min")
print(f"  count of combos    : {len(no_spurious)}")
