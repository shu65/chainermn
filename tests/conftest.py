import multiprocessing
import pytest


def dummy_func():
    pass


@pytest.fixture(scope='session', autouse=True)
def scope_session():
    multiprocessing.set_start_method('forkserver')
    # TODO make this silent
    p = multiprocessing.Process(target=dummy_func)
    p.start()
    p.join()
    yield
