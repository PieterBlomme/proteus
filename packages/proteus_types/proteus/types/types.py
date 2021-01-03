class BoundingBox:
    def __init__(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        class_name: str = None,
        score: float = None,
    ):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.class_name = class_name
        self.score = score


class Class:
    def __init__(self, class_name: str, score: float = None):
        self.class_name = class_name
        self.score = score


class Segmentation:
    def __init__(
        self,
        segmentation: list,
        class_name: str = None,
        score: float = None,
    ):
        self.segmentation = segmentation
        self.class_name = class_name
        self.score = score


class Coordinate:
    def __init__(self, name: str, x: int, y: int):
        self.name = name
        self.x = x
        self.y = y
