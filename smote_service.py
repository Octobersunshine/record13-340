import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from typing import Optional, Union, Tuple, Dict, Any


class SmoteOversamplingService:
    def __init__(
        self,
        method: str = "smote",
        sampling_strategy: Union[str, float, Dict[Any, int]] = "auto",
        random_state: int = 42,
        k_neighbors: int = 5,
        **kwargs
    ):
        self.method = method.lower()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.kwargs = kwargs
        self.oversampler = self._create_oversampler()
        self.class_distribution_before = None
        self.class_distribution_after = None

    def _create_oversampler(self):
        method_map = {
            "smote": SMOTE,
            "borderline": BorderlineSMOTE,
            "borderlinesmote": BorderlineSMOTE,
            "svm": SVMSMOTE,
            "svmsmote": SVMSMOTE,
            "adasyn": ADASYN,
        }

        if self.method not in method_map:
            raise ValueError(
                f"Unsupported method: {self.method}. "
                f"Supported methods: {list(method_map.keys())}"
            )

        oversampler_class = method_map[self.method]
        return oversampler_class(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors if self.method != "adasyn" else None,
            **self.kwargs
        )

    @staticmethod
    def _get_class_distribution(y: np.ndarray) -> Dict[Any, int]:
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))

    def analyze_distribution(
        self, y: Union[np.ndarray, pd.Series, list]
    ) -> Dict[str, Any]:
        y_array = np.array(y)
        distribution = self._get_class_distribution(y_array)
        total = sum(distribution.values())
        majority_class = max(distribution, key=distribution.get)
        minority_class = min(distribution, key=distribution.get)
        imbalance_ratio = distribution[majority_class] / distribution[minority_class]

        return {
            "distribution": distribution,
            "total_samples": total,
            "majority_class": majority_class,
            "majority_count": distribution[majority_class],
            "minority_class": minority_class,
            "minority_count": distribution[minority_class],
            "imbalance_ratio": round(imbalance_ratio, 2),
        }

    def fit_resample(
        self,
        X: Union[np.ndarray, pd.DataFrame, list],
        y: Union[np.ndarray, pd.Series, list],
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_array = np.array(X)
        y_array = np.array(y)

        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)

        self.class_distribution_before = self.analyze_distribution(y_array)

        X_resampled, y_resampled = self.oversampler.fit_resample(X_array, y_array)

        self.class_distribution_after = self.analyze_distribution(y_resampled)

        return X_resampled, y_resampled

    def fit_resample_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[list] = None,
    ) -> pd.DataFrame:
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]

        X = df[feature_columns].values
        y = df[target_column].values

        X_resampled, y_resampled = self.fit_resample(X, y)

        resampled_df = pd.DataFrame(X_resampled, columns=feature_columns)
        resampled_df[target_column] = y_resampled

        return resampled_df

    def get_summary(self) -> Dict[str, Any]:
        if self.class_distribution_before is None or self.class_distribution_after is None:
            raise ValueError(
                "No resampling has been performed yet. Call fit_resample() first."
            )

        before = self.class_distribution_before
        after = self.class_distribution_after

        samples_added = (
            after["total_samples"] - before["total_samples"]
        )

        return {
            "method": self.method,
            "before": before,
            "after": after,
            "samples_added": samples_added,
            "imbalance_ratio_before": before["imbalance_ratio"],
            "imbalance_ratio_after": after["imbalance_ratio"],
        }

    def print_summary(self):
        summary = self.get_summary()

        print("=" * 60)
        print(f"SMOTE Oversampling Summary (Method: {summary['method'].upper()})")
        print("=" * 60)

        print("\n--- Before Oversampling ---")
        before = summary["before"]
        for cls, count in before["distribution"].items():
            print(f"  Class {cls}: {count} samples")
        print(f"  Imbalance Ratio: {before['imbalance_ratio']}:1")
        print(f"  Total Samples: {before['total_samples']}")

        print("\n--- After Oversampling ---")
        after = summary["after"]
        for cls, count in after["distribution"].items():
            print(f"  Class {cls}: {count} samples")
        print(f"  Imbalance Ratio: {after['imbalance_ratio']}:1")
        print(f"  Total Samples: {after['total_samples']}")

        print("\n--- Changes ---")
        print(f"  Samples Added: {summary['samples_added']}")
        print(f"  Imbalance Ratio Improvement: "
              f"{summary['imbalance_ratio_before']} -> {summary['imbalance_ratio_after']}")
        print("=" * 60)


class CombinedSamplingService:
    def __init__(
        self,
        over_sampling_strategy: Union[str, float] = "auto",
        under_sampling_strategy: Union[str, float] = "auto",
        over_method: str = "smote",
        random_state: int = 42,
        k_neighbors: int = 5,
    ):
        self.over_sampling_strategy = over_sampling_strategy
        self.under_sampling_strategy = under_sampling_strategy
        self.over_method = over_method
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.pipeline = self._create_pipeline()
        self.class_distribution_before = None
        self.class_distribution_after = None

    def _create_pipeline(self):
        over = SmoteOversamplingService(
            method=self.over_method,
            sampling_strategy=self.over_sampling_strategy,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
        ).oversampler

        under = RandomUnderSampler(
            sampling_strategy=self.under_sampling_strategy,
            random_state=self.random_state,
        )

        return Pipeline([("over", over), ("under", under)])

    def fit_resample(
        self,
        X: Union[np.ndarray, pd.DataFrame, list],
        y: Union[np.ndarray, pd.Series, list],
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_array = np.array(X)
        y_array = np.array(y)

        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)

        self.class_distribution_before = SmoteOversamplingService._get_class_distribution(
            y_array
        )

        X_resampled, y_resampled = self.pipeline.fit_resample(X_array, y_array)

        self.class_distribution_after = SmoteOversamplingService._get_class_distribution(
            y_resampled
        )

        return X_resampled, y_resampled

    def get_summary(self) -> Dict[str, Any]:
        if self.class_distribution_before is None or self.class_distribution_after is None:
            raise ValueError(
                "No resampling has been performed yet. Call fit_resample() first."
            )

        before_total = sum(self.class_distribution_before.values())
        after_total = sum(self.class_distribution_after.values())

        return {
            "over_method": self.over_method,
            "before_distribution": self.class_distribution_before,
            "after_distribution": self.class_distribution_after,
            "samples_before": before_total,
            "samples_after": after_total,
            "samples_change": after_total - before_total,
        }

    def print_summary(self):
        summary = self.get_summary()

        print("=" * 60)
        print(f"Combined Sampling Summary (Over: {summary['over_method'].upper()})")
        print("=" * 60)

        print("\n--- Before Sampling ---")
        for cls, count in summary["before_distribution"].items():
            print(f"  Class {cls}: {count} samples")
        print(f"  Total Samples: {summary['samples_before']}")

        print("\n--- After Sampling ---")
        for cls, count in summary["after_distribution"].items():
            print(f"  Class {cls}: {count} samples")
        print(f"  Total Samples: {summary['samples_after']}")

        print("\n--- Changes ---")
        change = summary["samples_change"]
        print(f"  Sample Count Change: {change:+d}")
        print("=" * 60)
