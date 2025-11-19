from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

@dataclass
class TableColumn:
    bbox: List[Tuple[int, int, int, int]]                 # (x1, y1, x2, y2) column bounding box
    image: np.ndarray                               # cropped column image
    texts: List[str]   # OCR recognized strings
    boxes: np.ndarray
    # OCR bounding boxes per recognized text (aligned with `texts`)


@dataclass
class DetectedTable:
    page_index: int                                 # which page this table came from
    table_bbox: dict                               # full table bounding box {'x1':'332, 'y1':0, 'x2':'0, 'y2':4 }
    table_image: np.ndarray                         # cropped table image
    columns: List[TableColumn] = field(default_factory=list)

    @property
    def num_columns(self) -> int:
        return len(self.columns)
