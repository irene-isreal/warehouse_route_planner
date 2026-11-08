from dataclasses import dataclass


@dataclass
class SearchResult:
    path: list       # list of cells from start to goal; empty if no path exists
    explored: list   # cells in the order they were expanded
    time_ms: float

    @property
    def found(self):
        return len(self.path) > 0

    @property
    def cost(self):
        return len(self.path) - 1 if self.path else None


def reconstruct_path(parent, start, goal):
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return path[::-1]