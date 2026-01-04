"""
Metrics aggregation for benchmark experiments.
"""
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional
from statistics import mean, stdev

from ..models.benchmark import (
    ExperimentResult,
    TestResult,
    ExperimentMetrics,
)
from .storage import Storage


class MetricsCalculator:
    """Calculates aggregated metrics for experiments."""

    def __init__(self, storage: Storage):
        self.storage = storage

    async def calculate_metrics(self, experiment_id: str) -> ExperimentMetrics:
        """Calculate all metrics for an experiment."""
        # Get all results
        results = await self.storage.get_results_for_experiment(
            experiment_id, limit=10000
        )
        test_results = await self.storage.get_test_results_for_experiment(
            experiment_id
        )

        metrics = ExperimentMetrics(experiment_id=experiment_id)

        if not results:
            return metrics

        # Group results by different dimensions
        by_provider: Dict[str, List[ExperimentResult]] = defaultdict(list)
        by_tool_set: Dict[str, List[ExperimentResult]] = defaultdict(list)
        by_repo: Dict[str, List[ExperimentResult]] = defaultdict(list)

        for r in results:
            by_provider[r.provider].append(r)
            by_tool_set[r.tool_set].append(r)
            # Extract repo from instance_id (format: repo__issue)
            repo = r.instance_id.split('__')[0] if '__' in r.instance_id else 'unknown'
            by_repo[repo].append(r)

        # Calculate success rates
        metrics.overall_success_rate = self._success_rate(results)
        metrics.success_rate_by_provider = {
            p: self._success_rate(rs) for p, rs in by_provider.items()
        }
        metrics.success_rate_by_tool_set = {
            ts: self._success_rate(rs) for ts, rs in by_tool_set.items()
        }
        metrics.success_by_repo = {
            repo: self._success_rate(rs) for repo, rs in by_repo.items()
        }

        # Calculate tool usage patterns
        metrics.tool_usage_by_provider = self._tool_usage_by_provider(by_provider)
        metrics.avg_tool_calls_per_instance = {
            p: mean([r.tool_call_count for r in rs]) if rs else 0
            for p, rs in by_provider.items()
        }

        # Calculate token metrics
        metrics.avg_tokens_by_provider = {
            p: mean([r.total_tokens for r in rs if r.total_tokens]) if rs else 0
            for p, rs in by_provider.items()
        }
        metrics.avg_context_size_by_provider = {
            p: mean([r.context_size_tokens for r in rs if r.context_size_tokens]) if rs else 0
            for p, rs in by_provider.items()
        }

        # Calculate timing metrics
        metrics.avg_response_time_by_provider = {
            p: mean([r.response_time_seconds for r in rs if r.response_time_seconds]) if rs else 0
            for p, rs in by_provider.items()
        }

        # Calculate total runtime
        all_times = [r.response_time_seconds for r in results if r.response_time_seconds]
        metrics.total_runtime_seconds = sum(all_times)

        metrics.updated_at = datetime.utcnow()

        # Save metrics
        await self.storage.save_metrics(metrics)

        return metrics

    def _success_rate(self, results: List[ExperimentResult]) -> float:
        """Calculate success rate for a list of results."""
        if not results:
            return 0.0
        successes = sum(1 for r in results if r.success)
        return successes / len(results)

    def _tool_usage_by_provider(
        self,
        by_provider: Dict[str, List[ExperimentResult]]
    ) -> Dict[str, Dict[str, int]]:
        """Calculate tool usage counts by provider."""
        usage: Dict[str, Dict[str, int]] = {}

        for provider, results in by_provider.items():
            tool_counts: Dict[str, int] = defaultdict(int)
            for r in results:
                for tc in r.tool_calls:
                    tool_counts[tc.name] += 1
            usage[provider] = dict(tool_counts)

        return usage

    async def get_detailed_stats(self, experiment_id: str) -> Dict:
        """Get detailed statistics for an experiment."""
        results = await self.storage.get_results_for_experiment(
            experiment_id, limit=10000
        )
        test_results = await self.storage.get_test_results_for_experiment(
            experiment_id
        )

        # Build test result lookup
        test_by_result_id = {tr.result_id: tr for tr in test_results}

        stats = {
            "total_results": len(results),
            "successful_results": sum(1 for r in results if r.success),
            "failed_results": sum(1 for r in results if not r.success),
            "total_test_results": len(test_results),
            "resolved_count": sum(1 for tr in test_results if tr.resolved),
            "by_provider": {},
            "by_tool_set": {},
            "token_stats": {},
            "timing_stats": {},
            "tool_call_stats": {},
        }

        # Group by provider
        by_provider: Dict[str, List[ExperimentResult]] = defaultdict(list)
        for r in results:
            by_provider[r.provider].append(r)

        for provider, provider_results in by_provider.items():
            tokens = [r.total_tokens for r in provider_results if r.total_tokens]
            times = [r.response_time_seconds for r in provider_results if r.response_time_seconds]
            tool_calls = [r.tool_call_count for r in provider_results]

            provider_test_results = [
                test_by_result_id[r.id]
                for r in provider_results
                if r.id in test_by_result_id
            ]

            stats["by_provider"][provider] = {
                "total": len(provider_results),
                "successful": sum(1 for r in provider_results if r.success),
                "success_rate": self._success_rate(provider_results),
                "resolved": sum(1 for tr in provider_test_results if tr.resolved),
                "resolve_rate": (
                    sum(1 for tr in provider_test_results if tr.resolved) / len(provider_test_results)
                    if provider_test_results else 0
                ),
            }

            if tokens:
                stats["token_stats"][provider] = {
                    "mean": mean(tokens),
                    "min": min(tokens),
                    "max": max(tokens),
                    "std": stdev(tokens) if len(tokens) > 1 else 0,
                }

            if times:
                stats["timing_stats"][provider] = {
                    "mean": mean(times),
                    "min": min(times),
                    "max": max(times),
                    "std": stdev(times) if len(times) > 1 else 0,
                }

            if tool_calls:
                stats["tool_call_stats"][provider] = {
                    "mean": mean(tool_calls),
                    "min": min(tool_calls),
                    "max": max(tool_calls),
                    "total": sum(tool_calls),
                }

        # Group by tool set
        by_tool_set: Dict[str, List[ExperimentResult]] = defaultdict(list)
        for r in results:
            by_tool_set[r.tool_set].append(r)

        for tool_set, tool_set_results in by_tool_set.items():
            tool_set_test_results = [
                test_by_result_id[r.id]
                for r in tool_set_results
                if r.id in test_by_result_id
            ]

            stats["by_tool_set"][tool_set] = {
                "total": len(tool_set_results),
                "successful": sum(1 for r in tool_set_results if r.success),
                "success_rate": self._success_rate(tool_set_results),
                "resolved": sum(1 for tr in tool_set_test_results if tr.resolved),
                "resolve_rate": (
                    sum(1 for tr in tool_set_test_results if tr.resolved) / len(tool_set_test_results)
                    if tool_set_test_results else 0
                ),
            }

        return stats

    async def compare_providers(
        self,
        experiment_id: str
    ) -> Dict[str, Dict[str, float]]:
        """Generate provider comparison data for visualization."""
        metrics = await self.storage.get_metrics(experiment_id)
        if not metrics:
            metrics = await self.calculate_metrics(experiment_id)

        return {
            "success_rate": metrics.success_rate_by_provider,
            "avg_tokens": metrics.avg_tokens_by_provider,
            "avg_response_time": metrics.avg_response_time_by_provider,
            "avg_tool_calls": metrics.avg_tool_calls_per_instance,
        }

    async def get_tool_efficiency_analysis(
        self,
        experiment_id: str
    ) -> Dict:
        """Analyze tool efficiency (for thesis research)."""
        results = await self.storage.get_results_for_experiment(
            experiment_id, limit=10000
        )
        test_results = await self.storage.get_test_results_for_experiment(
            experiment_id
        )

        # Build test result lookup
        test_by_result_id = {tr.result_id: tr for tr in test_results}

        # Analyze correlation between tool usage and success
        tool_success_correlation = defaultdict(lambda: {"used": 0, "success": 0, "resolved": 0})

        for r in results:
            for tc in r.tool_calls:
                tool_success_correlation[tc.name]["used"] += 1
                if r.success:
                    tool_success_correlation[tc.name]["success"] += 1
                if r.id in test_by_result_id and test_by_result_id[r.id].resolved:
                    tool_success_correlation[tc.name]["resolved"] += 1

        # Calculate rates
        tool_analysis = {}
        for tool_name, counts in tool_success_correlation.items():
            tool_analysis[tool_name] = {
                "times_used": counts["used"],
                "success_when_used": counts["success"] / counts["used"] if counts["used"] > 0 else 0,
                "resolve_when_used": counts["resolved"] / counts["used"] if counts["used"] > 0 else 0,
            }

        # Analyze by tool set
        by_tool_set: Dict[str, List[ExperimentResult]] = defaultdict(list)
        for r in results:
            by_tool_set[r.tool_set].append(r)

        tool_set_analysis = {}
        for tool_set, tool_set_results in by_tool_set.items():
            tool_set_test_results = [
                test_by_result_id[r.id]
                for r in tool_set_results
                if r.id in test_by_result_id
            ]

            resolved = sum(1 for tr in tool_set_test_results if tr.resolved)

            tool_set_analysis[tool_set] = {
                "total_instances": len(tool_set_results),
                "avg_tool_calls": mean([r.tool_call_count for r in tool_set_results]),
                "success_rate": self._success_rate(tool_set_results),
                "resolve_rate": resolved / len(tool_set_test_results) if tool_set_test_results else 0,
            }

        return {
            "tool_correlation": tool_analysis,
            "tool_set_comparison": tool_set_analysis,
        }

    async def get_context_window_analysis(
        self,
        experiment_id: str
    ) -> Dict:
        """Analyze impact of context size on success (for thesis research)."""
        results = await self.storage.get_results_for_experiment(
            experiment_id, limit=10000
        )
        test_results = await self.storage.get_test_results_for_experiment(
            experiment_id
        )

        # Build test result lookup
        test_by_result_id = {tr.result_id: tr for tr in test_results}

        # Bucket results by context size
        context_buckets = defaultdict(list)
        bucket_size = 4000  # tokens

        for r in results:
            if r.context_size_tokens:
                bucket = (r.context_size_tokens // bucket_size) * bucket_size
                context_buckets[bucket].append(r)

        analysis = {}
        for bucket, bucket_results in sorted(context_buckets.items()):
            bucket_test_results = [
                test_by_result_id[r.id]
                for r in bucket_results
                if r.id in test_by_result_id
            ]

            resolved = sum(1 for tr in bucket_test_results if tr.resolved)

            analysis[f"{bucket}-{bucket + bucket_size}"] = {
                "count": len(bucket_results),
                "success_rate": self._success_rate(bucket_results),
                "resolve_rate": resolved / len(bucket_test_results) if bucket_test_results else 0,
                "avg_tokens": mean([r.total_tokens for r in bucket_results if r.total_tokens]),
            }

        return analysis
