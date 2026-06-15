import warnings
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, RandomOverSampler
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
            "random": RandomOverSampler,
            "randomoversampler": RandomOverSampler,
        }

        if self.method not in method_map:
            raise ValueError(
                f"Unsupported method: {self.method}. "
                f"Supported methods: {list(method_map.keys())}"
            )

        oversampler_class = method_map[self.method]

        if self.method in ("random", "randomoversampler"):
            return oversampler_class(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                **self.kwargs
            )

        if self.method == "adasyn":
            return oversampler_class(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_neighbors=self.k_neighbors,
                **self.kwargs
            )

        return oversampler_class(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
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

    def _get_minority_sample_count(self, y: np.ndarray) -> int:
        distribution = self._get_class_distribution(y)
        return min(distribution.values())

    def _adjust_k_neighbors(self, y: np.ndarray) -> int:
        if not self._is_knn_based_method():
            return 0

        n_minority = self._get_minority_sample_count(y)

        if self.k_neighbors < n_minority:
            return self.k_neighbors

        adjusted_k = max(1, n_minority - 1)

        if adjusted_k < self.k_neighbors:
            param_name = "n_neighbors" if self.method == "adasyn" else "k_neighbors"
            warnings.warn(
                f"少数类样本数 ({n_minority}) 少于 {param_name} ({self.k_neighbors})。"
                f"已自动将 {param_name} 调整为 {adjusted_k}。"
                f"建议增加少数类样本或减小 {param_name} 参数以获得更好的过采样效果。"
            )

        return adjusted_k

    def _is_knn_based_method(self) -> bool:
        return self.method in (
            "smote", "borderline", "borderlinesmote", "svm", "svmsmote", "adasyn"
        )

    def _validate_minority_samples(self, y: np.ndarray):
        n_minority = self._get_minority_sample_count(y)

        if n_minority < 1:
            raise ValueError("数据集中没有少数类样本，无法进行过采样。")

        if self._is_knn_based_method() and n_minority < 2:
            raise ValueError(
                f"{self.method.upper()} 过采样要求少数类至少有 2 个样本，"
                f"但当前少数类只有 {n_minority} 个样本。\n"
                f"建议方案：\n"
                f"  1. 收集更多少数类样本\n"
                f"  2. 使用 method='random' 随机重复采样代替（不要求样本数量）\n"
                f"  3. 检查数据标注是否正确，是否存在数据泄露或类别划分错误"
            )

    def _build_runtime_oversampler(self, y: np.ndarray):
        self._validate_minority_samples(y)

        if not self._is_knn_based_method():
            return self.oversampler

        adjusted_k = self._adjust_k_neighbors(y)

        if adjusted_k == self.k_neighbors and self.method != "adasyn":
            return self.oversampler

        method_map = {
            "smote": SMOTE,
            "borderline": BorderlineSMOTE,
            "borderlinesmote": BorderlineSMOTE,
            "svm": SVMSMOTE,
            "svmsmote": SVMSMOTE,
            "adasyn": ADASYN,
        }

        oversampler_class = method_map[self.method]

        if self.method == "adasyn":
            return oversampler_class(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_neighbors=adjusted_k,
                **{k: v for k, v in self.kwargs.items() if k != "n_neighbors"}
            )

        return oversampler_class(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            k_neighbors=adjusted_k,
            **{k: v for k, v in self.kwargs.items() if k != "k_neighbors"}
        )

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

        runtime_oversampler = self._build_runtime_oversampler(y_array)

        X_resampled, y_resampled = runtime_oversampler.fit_resample(X_array, y_array)

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
        self.class_distribution_before = None
        self.class_distribution_after = None
        self._smote_service = SmoteOversamplingService(
            method=self.over_method,
            sampling_strategy=self.over_sampling_strategy,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
        )
        self._under_sampler = RandomUnderSampler(
            sampling_strategy=self.under_sampling_strategy,
            random_state=self.random_state,
        )

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

        X_over, y_over = self._smote_service.fit_resample(X_array, y_array)

        X_resampled, y_resampled = self._under_sampler.fit_resample(X_over, y_over)

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
