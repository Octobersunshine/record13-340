import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from smote_service import SmoteOversamplingService, CombinedSamplingService


def example_basic_usage():
    print("=" * 60)
    print("Example 1: Basic SMOTE Oversampling")
    print("=" * 60)

    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42,
    )

    print(f"Original dataset shape: {X.shape}")
    print(f"Original class distribution: {np.bincount(y)}")

    service = SmoteOversamplingService(
        method="smote",
        sampling_strategy="auto",
        random_state=42,
        k_neighbors=5,
    )

    X_resampled, y_resampled = service.fit_resample(X, y)

    print(f"\nResampled dataset shape: {X_resampled.shape}")
    print(f"Resampled class distribution: {np.bincount(y_resampled)}")

    service.print_summary()


def example_dataframe_usage():
    print("\n" + "=" * 60)
    print("Example 2: DataFrame Usage")
    print("=" * 60)

    np.random.seed(42)
    n_majority = 900
    n_minority = 100

    df_majority = pd.DataFrame({
        "feature1": np.random.normal(0, 1, n_majority),
        "feature2": np.random.normal(0, 1, n_majority),
        "feature3": np.random.normal(0, 1, n_majority),
        "label": 0,
    })

    df_minority = pd.DataFrame({
        "feature1": np.random.normal(2, 1, n_minority),
        "feature2": np.random.normal(2, 1, n_minority),
        "feature3": np.random.normal(2, 1, n_minority),
        "label": 1,
    })

    df = pd.concat([df_majority, df_minority], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Original DataFrame shape: {df.shape}")
    print(f"Original class distribution:\n{df['label'].value_counts()}")

    service = SmoteOversamplingService(method="smote", random_state=42)
    resampled_df = service.fit_resample_dataframe(df, target_column="label")

    print(f"\nResampled DataFrame shape: {resampled_df.shape}")
    print(f"Resampled class distribution:\n{resampled_df['label'].value_counts()}")


def example_different_methods():
    print("\n" + "=" * 60)
    print("Example 3: Different Oversampling Methods")
    print("=" * 60)

    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_classes=2,
        weights=[0.85, 0.15],
        random_state=42,
    )

    methods = ["smote", "borderline", "svm", "adasyn", "random"]

    for method in methods:
        print(f"\n--- Method: {method.upper()} ---")
        try:
            service = SmoteOversamplingService(
                method=method,
                random_state=42,
            )
            X_res, y_res = service.fit_resample(X, y)
            print(f"Before: {dict(zip(*np.unique(y, return_counts=True)))}")
            print(f"After:  {dict(zip(*np.unique(y_res, return_counts=True)))}")
        except Exception as e:
            print(f"Error: {e}")


def example_random_single_sample():
    print("\n" + "=" * 60)
    print("Example 7: Random Oversampling with Few Samples")
    print("=" * 60)

    X = np.array([
        [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
        [100, 100]
    ])
    y = np.array([0, 0, 0, 0, 0, 1])

    print(f"Original distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print("\nTrying SMOTE (should fail with helpful message):")
    try:
        service = SmoteOversamplingService(method="smote", random_state=42)
        service.fit_resample(X, y)
    except ValueError as e:
        print(f"  Error: {str(e).split(chr(10))[0]}")

    print("\nUsing Random Oversampling (works with 1 sample):")
    service = SmoteOversamplingService(method="random", random_state=42)
    X_res, y_res = service.fit_resample(X, y)
    print(f"  Result: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    service.print_summary()


def example_custom_sampling_strategy():
    print("\n" + "=" * 60)
    print("Example 4: Custom Sampling Strategy")
    print("=" * 60)

    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42,
    )

    print(f"Original: {dict(zip(*np.unique(y, return_counts=True)))}")

    service = SmoteOversamplingService(
        method="smote",
        sampling_strategy=0.5,
        random_state=42,
    )
    X_res, y_res = service.fit_resample(X, y)

    print(f"After (sampling_strategy=0.5): {dict(zip(*np.unique(y_res, return_counts=True)))}")
    print(f"Minority/Majority ratio: {np.sum(y_res == 1) / np.sum(y_res == 0):.2f}")


def example_combined_sampling():
    print("\n" + "=" * 60)
    print("Example 5: Combined Over- and Under-sampling")
    print("=" * 60)

    X, y = make_classification(
        n_samples=2000,
        n_features=5,
        n_classes=2,
        weights=[0.95, 0.05],
        random_state=42,
    )

    print(f"Original: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Original total: {len(y)}")

    service = CombinedSamplingService(
        over_sampling_strategy=0.3,
        under_sampling_strategy=0.7,
        over_method="smote",
        random_state=42,
    )

    X_res, y_res = service.fit_resample(X, y)

    print(f"After combined: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    print(f"After total: {len(y_res)}")

    service.print_summary()


def example_distribution_analysis():
    print("\n" + "=" * 60)
    print("Example 6: Distribution Analysis")
    print("=" * 60)

    y = np.array([0] * 950 + [1] * 50)

    service = SmoteOversamplingService()
    analysis = service.analyze_distribution(y)

    print("Distribution Analysis:")
    for key, value in analysis.items():
        if key == "distribution":
            print(f"  {key}:")
            for cls, count in value.items():
                print(f"    Class {cls}: {count}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    example_basic_usage()
    example_dataframe_usage()
    example_different_methods()
    example_custom_sampling_strategy()
    example_combined_sampling()
    example_distribution_analysis()
    example_random_single_sample()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
