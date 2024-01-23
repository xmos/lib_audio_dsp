
from .composite_stage import CompositeStage

class Thread(CompositeStage):
    """
    A composite stage used to represent a thread in the pipeline. Create
    using Pipeline.thread rather than instantiating directly.

    Parameters
    ----------
    id : int
        Thread index
    kwargs : dict
        forwarded to __init__ of CompositeStage

    Attributes
    ----------
    id : int
        Thread index
    """
    def __init__(self, id: int, **kwargs):
        super().__init__(name = f"Thread {id}", **kwargs)
        self.id = id

    def __enter__(self):
        """Support for context manager"""
        return self

    def __exit__(self ,type, value, traceback):
        """Support for context manager"""
        ...

        
