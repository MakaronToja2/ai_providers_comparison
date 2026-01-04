"""
SQLite storage layer for benchmark experiments and results.
Uses aiosqlite for async database operations.
"""
import json
import aiosqlite
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..models.benchmark import (
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    ExperimentResult,
    TestResult,
    ExperimentMetrics,
    ToolCallRecord,
)


class Storage:
    """Async SQLite storage for benchmark data."""

    # Class-level flag to track initialization across all instances
    _global_initialized = False
    # Store the absolute path to avoid issues with working directory changes
    _absolute_db_path = None

    def __init__(self, db_path: str = "benchmark.db"):
        # Convert to absolute path on first initialization
        if Storage._absolute_db_path is None:
            Storage._absolute_db_path = Path(db_path).resolve()
        self.db_path = Storage._absolute_db_path

    @asynccontextmanager
    async def _get_db(self):
        """Get database connection with row factory. Auto-initializes on first use."""
        # Auto-initialize if not done yet
        if not Storage._global_initialized:
            await self._do_initialize()

        async with aiosqlite.connect(self.db_path, timeout=30.0) as db:
            db.row_factory = aiosqlite.Row
            # Enable WAL mode for better concurrent write handling
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=30000")
            yield db

    async def _do_initialize(self):
        """Actually perform initialization."""
        if Storage._global_initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await self._create_tables(db)
            await db.commit()

        Storage._global_initialized = True
        print("Database tables created successfully")

    async def _create_tables(self, db):
        """Create all database tables."""
        # Experiments table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                config TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                total_instances INTEGER DEFAULT 0,
                completed_instances INTEGER DEFAULT 0
            )
        """)

        # Experiment results table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS experiment_results (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                instance_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                tool_set TEXT NOT NULL,
                response_content TEXT,
                generated_patch TEXT,
                raw_response TEXT,
                success INTEGER DEFAULT 0,
                error_message TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                context_size_chars INTEGER,
                context_size_tokens INTEGER,
                response_time_seconds REAL,
                tool_calls TEXT,
                tool_call_count INTEGER DEFAULT 0,
                successful_tool_calls INTEGER DEFAULT 0,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)

        # Test results table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id TEXT PRIMARY KEY,
                result_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                instance_id TEXT NOT NULL,
                patch_applied INTEGER DEFAULT 0,
                patch_error TEXT,
                fail_to_pass_total INTEGER DEFAULT 0,
                fail_to_pass_passed INTEGER DEFAULT 0,
                pass_to_pass_total INTEGER DEFAULT 0,
                pass_to_pass_passed INTEGER DEFAULT 0,
                resolved INTEGER DEFAULT 0,
                test_output TEXT,
                execution_time_seconds REAL,
                created_at TEXT,
                FOREIGN KEY (result_id) REFERENCES experiment_results(id),
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)

        # Experiment metrics table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS experiment_metrics (
                experiment_id TEXT PRIMARY KEY,
                overall_success_rate REAL,
                success_rate_by_provider TEXT,
                success_rate_by_tool_set TEXT,
                tool_usage_by_provider TEXT,
                avg_tool_calls_per_instance TEXT,
                avg_tokens_by_provider TEXT,
                avg_context_size_by_provider TEXT,
                avg_response_time_by_provider TEXT,
                success_by_repo TEXT,
                total_runtime_seconds REAL,
                updated_at TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)

        # Create indexes
        await db.execute("CREATE INDEX IF NOT EXISTS idx_results_experiment ON experiment_results(experiment_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_results_provider ON experiment_results(provider)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_test_results_experiment ON test_results(experiment_id)")

    async def initialize(self):
        """Initialize database schema (for backward compatibility)."""
        await self._do_initialize()

    # ==================== Experiments ====================

    async def create_experiment(self, experiment: Experiment) -> Experiment:
        """Create a new experiment."""
        async with self._get_db() as db:
            await db.execute("""
                INSERT INTO experiments (id, name, description, status, config, created_at,
                                        started_at, completed_at, total_instances, completed_instances)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment.id,
                experiment.name,
                experiment.description,
                experiment.status.value,
                experiment.config.model_dump_json(),
                experiment.created_at.isoformat(),
                experiment.started_at.isoformat() if experiment.started_at else None,
                experiment.completed_at.isoformat() if experiment.completed_at else None,
                experiment.total_instances,
                experiment.completed_instances,
            ))
            await db.commit()
        return experiment

    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        async with self._get_db() as db:
            cursor = await db.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (experiment_id,)
            )
            row = await cursor.fetchone()
            if row:
                return self._row_to_experiment(row)
        return None

    async def list_experiments(self, limit: int = 100, offset: int = 0) -> List[Experiment]:
        """List all experiments."""
        async with self._get_db() as db:
            cursor = await db.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
            rows = await cursor.fetchall()
            return [self._row_to_experiment(row) for row in rows]

    async def update_experiment(self, experiment: Experiment) -> None:
        """Update an experiment."""
        async with self._get_db() as db:
            await db.execute("""
                UPDATE experiments SET
                    name = ?,
                    description = ?,
                    status = ?,
                    config = ?,
                    started_at = ?,
                    completed_at = ?,
                    total_instances = ?,
                    completed_instances = ?
                WHERE id = ?
            """, (
                experiment.name,
                experiment.description,
                experiment.status.value,
                experiment.config.model_dump_json(),
                experiment.started_at.isoformat() if experiment.started_at else None,
                experiment.completed_at.isoformat() if experiment.completed_at else None,
                experiment.total_instances,
                experiment.completed_instances,
                experiment.id,
            ))
            await db.commit()

    async def delete_experiment(self, experiment_id: str) -> None:
        """Delete experiment and all related data."""
        async with self._get_db() as db:
            # Delete in order due to foreign keys
            await db.execute("DELETE FROM test_results WHERE experiment_id = ?", (experiment_id,))
            await db.execute("DELETE FROM experiment_results WHERE experiment_id = ?", (experiment_id,))
            await db.execute("DELETE FROM experiment_metrics WHERE experiment_id = ?", (experiment_id,))
            await db.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
            await db.commit()

    def _row_to_experiment(self, row: aiosqlite.Row) -> Experiment:
        """Convert database row to Experiment model."""
        return Experiment(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            status=ExperimentStatus(row["status"]),
            config=ExperimentConfig.model_validate_json(row["config"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            total_instances=row["total_instances"],
            completed_instances=row["completed_instances"],
        )

    # ==================== Results ====================

    async def save_result(self, result: ExperimentResult) -> None:
        """Save or update an experiment result."""
        async with self._get_db() as db:
            tool_calls_json = json.dumps([tc.model_dump() for tc in result.tool_calls])
            raw_response_json = json.dumps(result.raw_response) if result.raw_response else None

            await db.execute("""
                INSERT OR REPLACE INTO experiment_results
                (id, experiment_id, instance_id, provider, model, tool_set,
                 response_content, generated_patch, raw_response,
                 response_time_seconds, started_at, completed_at,
                 prompt_tokens, completion_tokens, total_tokens,
                 tool_calls, tool_call_count, successful_tool_calls,
                 context_size_chars, context_size_tokens, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.id,
                result.experiment_id,
                result.instance_id,
                result.provider,
                result.model,
                result.tool_set,
                result.response_content,
                result.generated_patch,
                raw_response_json,
                result.response_time_seconds,
                result.started_at.isoformat() if result.started_at else None,
                result.completed_at.isoformat() if result.completed_at else None,
                result.prompt_tokens,
                result.completion_tokens,
                result.total_tokens,
                tool_calls_json,
                result.tool_call_count,
                result.successful_tool_calls,
                result.context_size_chars,
                result.context_size_tokens,
                1 if result.success else 0,
                result.error_message,
            ))
            await db.commit()

    async def get_result(self, result_id: str) -> Optional[ExperimentResult]:
        """Get a single result by ID."""
        async with self._get_db() as db:
            cursor = await db.execute(
                "SELECT * FROM experiment_results WHERE id = ?",
                (result_id,)
            )
            row = await cursor.fetchone()
            if row:
                return self._row_to_result(row)
        return None

    async def get_results_for_experiment(
        self,
        experiment_id: str,
        provider: Optional[str] = None,
        success_only: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> List[ExperimentResult]:
        """Get results for an experiment with optional filters."""
        async with self._get_db() as db:
            query = "SELECT * FROM experiment_results WHERE experiment_id = ?"
            params: List[Any] = [experiment_id]

            if provider:
                query += " AND provider = ?"
                params.append(provider)

            if success_only:
                query += " AND success = 1"

            query += " ORDER BY completed_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            return [self._row_to_result(row) for row in rows]

    async def get_completed_result_keys(self, experiment_id: str) -> set:
        """Get set of (instance_id, provider, model, tool_set) for completed results."""
        async with self._get_db() as db:
            cursor = await db.execute("""
                SELECT instance_id, provider, model, tool_set
                FROM experiment_results
                WHERE experiment_id = ? AND success = 1
            """, (experiment_id,))
            rows = await cursor.fetchall()
            return {(r["instance_id"], r["provider"], r["model"], r["tool_set"]) for r in rows}

    async def count_results(self, experiment_id: str, success_only: bool = False) -> int:
        """Count results for an experiment."""
        async with self._get_db() as db:
            query = "SELECT COUNT(*) as count FROM experiment_results WHERE experiment_id = ?"
            if success_only:
                query += " AND success = 1"
            cursor = await db.execute(query, (experiment_id,))
            row = await cursor.fetchone()
            return row["count"] if row else 0

    def _row_to_result(self, row: aiosqlite.Row) -> ExperimentResult:
        """Convert database row to ExperimentResult model."""
        tool_calls = []
        if row["tool_calls"]:
            tool_calls_data = json.loads(row["tool_calls"])
            tool_calls = [ToolCallRecord(**tc) for tc in tool_calls_data]

        raw_response = None
        if row["raw_response"]:
            raw_response = json.loads(row["raw_response"])

        return ExperimentResult(
            id=row["id"],
            experiment_id=row["experiment_id"],
            instance_id=row["instance_id"],
            provider=row["provider"],
            model=row["model"],
            tool_set=row["tool_set"],
            response_content=row["response_content"],
            generated_patch=row["generated_patch"],
            raw_response=raw_response,
            response_time_seconds=row["response_time_seconds"],
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
            tool_calls=tool_calls,
            tool_call_count=row["tool_call_count"],
            successful_tool_calls=row["successful_tool_calls"],
            context_size_chars=row["context_size_chars"],
            context_size_tokens=row["context_size_tokens"],
            success=bool(row["success"]),
            error_message=row["error_message"],
        )

    # ==================== Test Results ====================

    async def save_test_result(self, test_result: TestResult) -> None:
        """Save a test result."""
        async with self._get_db() as db:
            await db.execute("""
                INSERT OR REPLACE INTO test_results
                (id, result_id, experiment_id, instance_id,
                 patch_applied, patch_error,
                 fail_to_pass_total, fail_to_pass_passed,
                 pass_to_pass_total, pass_to_pass_passed,
                 resolved, test_output, execution_time_seconds, executed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_result.id,
                test_result.result_id,
                test_result.experiment_id,
                test_result.instance_id,
                1 if test_result.patch_applied else 0,
                test_result.patch_error,
                test_result.fail_to_pass_total,
                test_result.fail_to_pass_passed,
                test_result.pass_to_pass_total,
                test_result.pass_to_pass_passed,
                1 if test_result.resolved else 0,
                test_result.test_output,
                test_result.execution_time_seconds,
                test_result.executed_at.isoformat() if test_result.executed_at else None,
            ))
            await db.commit()

    async def get_test_results_for_experiment(
        self,
        experiment_id: str,
        resolved_only: bool = False
    ) -> List[TestResult]:
        """Get test results for an experiment."""
        async with self._get_db() as db:
            query = "SELECT * FROM test_results WHERE experiment_id = ?"
            if resolved_only:
                query += " AND resolved = 1"

            cursor = await db.execute(query, (experiment_id,))
            rows = await cursor.fetchall()
            return [self._row_to_test_result(row) for row in rows]

    async def count_test_results(self, experiment_id: str, resolved_only: bool = False) -> int:
        """Count test results for an experiment."""
        async with self._get_db() as db:
            query = "SELECT COUNT(*) as count FROM test_results WHERE experiment_id = ?"
            if resolved_only:
                query += " AND resolved = 1"
            cursor = await db.execute(query, (experiment_id,))
            row = await cursor.fetchone()
            return row["count"] if row else 0

    def _row_to_test_result(self, row: aiosqlite.Row) -> TestResult:
        """Convert database row to TestResult model."""
        return TestResult(
            id=row["id"],
            result_id=row["result_id"],
            experiment_id=row["experiment_id"],
            instance_id=row["instance_id"],
            patch_applied=bool(row["patch_applied"]),
            patch_error=row["patch_error"],
            fail_to_pass_total=row["fail_to_pass_total"],
            fail_to_pass_passed=row["fail_to_pass_passed"],
            pass_to_pass_total=row["pass_to_pass_total"],
            pass_to_pass_passed=row["pass_to_pass_passed"],
            resolved=bool(row["resolved"]),
            test_output=row["test_output"],
            execution_time_seconds=row["execution_time_seconds"],
            executed_at=datetime.fromisoformat(row["executed_at"]) if row["executed_at"] else None,
        )

    # ==================== Metrics ====================

    async def save_metrics(self, metrics: ExperimentMetrics) -> None:
        """Save or update experiment metrics."""
        async with self._get_db() as db:
            await db.execute("""
                INSERT OR REPLACE INTO experiment_metrics
                (experiment_id, overall_success_rate, success_rate_by_provider,
                 success_rate_by_tool_set, tool_usage_by_provider, avg_tool_calls_per_instance,
                 avg_tokens_by_provider, avg_context_size_by_provider,
                 avg_response_time_by_provider, total_runtime_seconds,
                 success_by_repo, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.experiment_id,
                metrics.overall_success_rate,
                json.dumps(metrics.success_rate_by_provider),
                json.dumps(metrics.success_rate_by_tool_set),
                json.dumps(metrics.tool_usage_by_provider),
                json.dumps(metrics.avg_tool_calls_per_instance),
                json.dumps(metrics.avg_tokens_by_provider),
                json.dumps(metrics.avg_context_size_by_provider),
                json.dumps(metrics.avg_response_time_by_provider),
                metrics.total_runtime_seconds,
                json.dumps(metrics.success_by_repo),
                metrics.updated_at.isoformat(),
            ))
            await db.commit()

    async def get_metrics(self, experiment_id: str) -> Optional[ExperimentMetrics]:
        """Get metrics for an experiment."""
        async with self._get_db() as db:
            cursor = await db.execute(
                "SELECT * FROM experiment_metrics WHERE experiment_id = ?",
                (experiment_id,)
            )
            row = await cursor.fetchone()
            if row:
                return self._row_to_metrics(row)
        return None

    def _row_to_metrics(self, row: aiosqlite.Row) -> ExperimentMetrics:
        """Convert database row to ExperimentMetrics model."""
        return ExperimentMetrics(
            experiment_id=row["experiment_id"],
            overall_success_rate=row["overall_success_rate"] or 0.0,
            success_rate_by_provider=json.loads(row["success_rate_by_provider"] or "{}"),
            success_rate_by_tool_set=json.loads(row["success_rate_by_tool_set"] or "{}"),
            tool_usage_by_provider=json.loads(row["tool_usage_by_provider"] or "{}"),
            avg_tool_calls_per_instance=json.loads(row["avg_tool_calls_per_instance"] or "{}"),
            avg_tokens_by_provider=json.loads(row["avg_tokens_by_provider"] or "{}"),
            avg_context_size_by_provider=json.loads(row["avg_context_size_by_provider"] or "{}"),
            avg_response_time_by_provider=json.loads(row["avg_response_time_by_provider"] or "{}"),
            total_runtime_seconds=row["total_runtime_seconds"] or 0.0,
            success_by_repo=json.loads(row["success_by_repo"] or "{}"),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.utcnow(),
        )

    # ==================== Export ====================

    async def export_results_to_dict(self, experiment_id: str) -> Dict[str, Any]:
        """Export all experiment data as a dictionary for JSON/CSV export."""
        experiment = await self.get_experiment(experiment_id)
        if not experiment:
            return {}

        results = await self.get_results_for_experiment(experiment_id, limit=10000)
        test_results = await self.get_test_results_for_experiment(experiment_id)
        metrics = await self.get_metrics(experiment_id)

        return {
            "experiment": experiment.model_dump(),
            "results": [r.model_dump() for r in results],
            "test_results": [tr.model_dump() for tr in test_results],
            "metrics": metrics.model_dump() if metrics else None,
        }
