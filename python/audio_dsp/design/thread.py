
from .composite_stage import CompositeStage

class Thread(CompositeStage):
    def __init__(self, id: int, **kwargs):
        super().__init__(name = f"Thread {id}", **kwargs)
        self.id = id

    def __enter__(self):
        return self

    def __exit__(self ,type, value, traceback):
        ...

        
