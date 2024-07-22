


from pathlib import Path
import pytest
import sys
from subprocess import run

EXAMPLES_DIR = Path(__file__).parent/"doc_examples"


EXAMPLES = list(EXAMPLES_DIR.glob("*.py"))

@pytest.mark.parametrize("example", EXAMPLES, ids=[e.name for e in EXAMPLES])
def test_doc_examples(example):
    """Run all the python scripts in doc_examples/"""
    run([sys.executable, example], check=True)
    
    
