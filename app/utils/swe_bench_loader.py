import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging

from ..models.swe_bench import SWEBenchInstance, SWEBenchDatasetInfo

logger = logging.getLogger(__name__)


class SWEBenchLoader:
    """Utility class for loading and managing SWE-bench dataset"""
    
    def __init__(self):
        self._dataset_cache: Dict[str, pd.DataFrame] = {}
        self._load_times: Dict[str, datetime] = {}
    
    def load_dataset(self, split: str = "dev") -> SWEBenchDatasetInfo:
        """
        Load SWE-bench dataset split
        
        Args:
            split: Dataset split to load ('dev' or 'test')
            
        Returns:
            Dataset information
        """
        if split not in ["dev", "test"]:
            raise ValueError("Split must be 'dev' or 'test'")
        
        try:
            # Use huggingface datasets path format
            splits_mapping = {
                'dev': 'data/dev-00000-of-00001.parquet',
                'test': 'data/test-00000-of-00001.parquet'
            }
            
            logger.info(f"Loading SWE-bench {split} dataset...")
            df = pd.read_parquet(f"hf://datasets/SWE-bench/SWE-bench_Lite/{splits_mapping[split]}")
            
            # Cache the dataset
            self._dataset_cache[split] = df
            self._load_times[split] = datetime.now()
            
            # Get sample instance IDs (first 5)
            sample_ids = df['instance_id'].head().tolist() if 'instance_id' in df.columns else []
            
            logger.info(f"Successfully loaded {len(df)} instances from SWE-bench {split}")
            
            return SWEBenchDatasetInfo(
                split=split,
                total_instances=len(df),
                sample_instance_ids=sample_ids,
                loaded_at=self._load_times[split]
            )
            
        except Exception as e:
            logger.error(f"Failed to load SWE-bench dataset {split}: {str(e)}")
            raise RuntimeError(f"Failed to load SWE-bench dataset: {str(e)}")
    
    def get_instance(self, instance_id: str, split: str = "dev") -> Optional[SWEBenchInstance]:
        """
        Get a specific instance by ID
        
        Args:
            instance_id: Instance ID to retrieve
            split: Dataset split to search in
            
        Returns:
            SWE-bench instance or None if not found
        """
        # Load dataset if not cached
        if split not in self._dataset_cache:
            self.load_dataset(split)
        
        df = self._dataset_cache[split]
        
        # Find the instance
        instance_rows = df[df['instance_id'] == instance_id]
        
        if instance_rows.empty:
            logger.warning(f"Instance {instance_id} not found in {split} split")
            return None
        
        # Convert to our model
        row = instance_rows.iloc[0]
        
        try:
            instance = SWEBenchInstance(
                instance_id=row.get('instance_id', ''),
                repo=row.get('repo', ''),
                base_commit=row.get('base_commit', ''),
                patch=row.get('patch', ''),
                test_patch=row.get('test_patch', ''),
                problem_statement=row.get('problem_statement', ''),
                hints_text=row.get('hints_text'),
                created_at=row.get('created_at'),
                version=row.get('version'),
                FAIL_TO_PASS=row.get('FAIL_TO_PASS'),
                PASS_TO_PASS=row.get('PASS_TO_PASS'),
                environment_setup_commit=row.get('environment_setup_commit')
            )
            
            logger.info(f"Retrieved instance {instance_id} from {split} split")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to parse instance {instance_id}: {str(e)}")
            return None
    
    def list_instances(self, split: str = "dev", limit: int = 10) -> List[str]:
        """
        List instance IDs from the dataset
        
        Args:
            split: Dataset split to list from
            limit: Maximum number of IDs to return
            
        Returns:
            List of instance IDs
        """
        # Load dataset if not cached
        if split not in self._dataset_cache:
            self.load_dataset(split)
        
        df = self._dataset_cache[split]
        
        if 'instance_id' not in df.columns:
            logger.warning(f"No instance_id column found in {split} dataset")
            return []
        
        return df['instance_id'].head(limit).tolist()
    
    def get_dataset_info(self, split: str = "dev") -> Optional[SWEBenchDatasetInfo]:
        """Get information about a loaded dataset split"""
        if split not in self._dataset_cache:
            return None
        
        df = self._dataset_cache[split]
        sample_ids = df['instance_id'].head().tolist() if 'instance_id' in df.columns else []
        
        return SWEBenchDatasetInfo(
            split=split,
            total_instances=len(df),
            sample_instance_ids=sample_ids,
            loaded_at=self._load_times.get(split, datetime.now())
        )
    
    def search_instances(self, split: str = "dev", repo: Optional[str] = None, 
                        limit: int = 10) -> List[str]:
        """
        Search for instances with optional filtering
        
        Args:
            split: Dataset split to search
            repo: Filter by repository name
            limit: Maximum results to return
            
        Returns:
            List of matching instance IDs
        """
        # Load dataset if not cached
        if split not in self._dataset_cache:
            self.load_dataset(split)
        
        df = self._dataset_cache[split]
        
        # Apply filters
        if repo:
            df = df[df['repo'].str.contains(repo, case=False, na=False)]
        
        if 'instance_id' not in df.columns:
            return []
        
        return df['instance_id'].head(limit).tolist()


# Global instance for reuse
swe_bench_loader = SWEBenchLoader()