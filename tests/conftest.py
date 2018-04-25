import multiprocessing
import pytest


@pytest.fixture(scope='session', autouse=True)
def scope_session():
    print("setup before session")
    multiprocessing.set_start_method('forkserver')
    # TODO make this silent
    p = multiprocessing.Process(target=print, args=('Initialize forkserver',))
    p.start()
    p.join()
    yield
    print("teardown after session")
