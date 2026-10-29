import pytest

from predlab.db.repo import init_market_db, init_sim_db, make_engine


@pytest.fixture
def market_engine(tmp_path):
    eng = make_engine(f"sqlite:///{tmp_path / 'market.db'}")
    init_market_db(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def sim_engine(tmp_path):
    eng = make_engine(f"sqlite:///{tmp_path / 'sim.db'}")
    init_sim_db(eng)
    yield eng
    eng.dispose()


class FakeSleep:
    def __init__(self):
        self.calls = []

    async def __call__(self, s):
        self.calls.append(s)


@pytest.fixture
def fake_sleep():
    return FakeSleep()
