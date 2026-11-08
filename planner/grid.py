class Grid:
    """4-connected grid. Cells are (row, col). Blocked cells are shelves/obstacles."""

    def __init__(self, rows, cols, blocked=None):
        self.rows = rows
        self.cols = cols
        self.blocked = set(blocked or [])

    @classmethod
    def from_ascii(cls, lines):
        """Build a grid from text: '#' blocked, 'S' start, 'G' goal, anything else free."""
        blocked, start, goal = set(), None, None
        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                if ch == "#":
                    blocked.add((r, c))
                elif ch == "S":
                    start = (r, c)
                elif ch == "G":
                    goal = (r, c)
        return cls(len(lines), len(lines[0]), blocked), start, goal

    def in_bounds(self, cell):
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, cell):
        return self.in_bounds(cell) and cell not in self.blocked

    def neighbors(self, cell):
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if self.is_free(nxt):
                yield nxt