
from python import run_pipeline_xcoreai


def pytest_configure(config):
    run_pipeline_xcoreai.FORCE_ADAPTER_ID = config.getoption("--adapter-id")

def pytest_addoption(parser):
    parser.addoption(
        "--adapter-id", action="store", default=None, help="Force tests to use specific adapter"
    )
