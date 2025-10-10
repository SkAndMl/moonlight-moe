import re
from typing import Optional, Tuple
import datasets
from .base import MultiTaskDatasetProcessor


class TinyStoriesProcessor(MultiTaskDatasetProcessor):
    
    DATASET_ID = "skeskinen/TinyStories-Instruct-hf"
    SPLIT = "train"
    
    def load_dataset(self) -> None:
        self.dataset = datasets.load_dataset(
            self.DATASET_ID,
            split=self.SPLIT,
            token=self.token
        )
        self._size = len(self.dataset)
    
    def extract_sample(self, idx: int) -> Optional[Tuple[str, str, str]]:
        if not self.is_loaded():
            self.load_dataset()
        
        try:
            text = self.dataset[idx]["text"]
            other_sections, story_section = self._parse_sections(text, "Story")
            
            if len(other_sections) == 1 and "Summary:" in other_sections[0]:
                user_content = story_section  # Just the story text
                assistant_content = other_sections[0]  # The summary
                task_type = "summarize"
                
            else:
                other_sections = [sec for sec in other_sections if "Summary" not in sec]
                
                if not other_sections:
                    return None
                
                user_content = "\n".join(other_sections)
                assistant_content = story_section
                task_type = "generate_long"
            
            return user_content, assistant_content, task_type
            
        except (KeyError, IndexError, Exception):
            return None
    
    def _parse_sections(self, text: str, separate_section: str) -> Tuple[list, str]:
        section_pattern = r'(\w+):\s*(.*?)(?=\n\w+:|$)'
        sections = re.findall(section_pattern, text, re.DOTALL)
        
        story_section = ""
        other_sections = []
        
        for section_name, content in sections:
            if section_name == separate_section:
                story_section = f"{section_name}: {content.strip()}"
            else:
                other_sections.append(f"{section_name}: {content.strip()}")
        
        return other_sections, story_section