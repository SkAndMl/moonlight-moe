from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

class BaseDatasetProcessor(ABC):    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self.dataset = None
        self._size = None
    
    @abstractmethod
    def load_dataset(self) -> None:
        pass
    
    @abstractmethod
    def extract_sample(self, idx: int) -> Optional[Tuple[Any, str]]:
        pass
    
    @abstractmethod
    def get_task_type(self) -> str:
        pass
    
    def get_dataset_size(self) -> int:
        if self._size is None:
            if self.dataset is None:
                self.load_dataset()
            self._size = len(self.dataset)
        return self._size
    
    def is_loaded(self) -> bool:
        return self.dataset is not None
    
    def __len__(self) -> int:
        return self.get_dataset_size()


class MultiTaskDatasetProcessor(BaseDatasetProcessor):
    @abstractmethod
    def extract_sample(self, idx: int) -> Optional[Tuple[Any, str, str]]:
        pass
    
    def get_task_type(self) -> str:
        raise NotImplementedError("MultiTaskDatasetProcessor does not have a single task type")