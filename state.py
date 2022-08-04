from typing import FrozenSet, List

class PartialCover:

    def __init__(self, cur_edges: List[FrozenSet[int]], cur_vertices: List[int]):
        self._edges = cur_edges
        self._vertices = cur_vertices
