"""The data ledger refuses claims that would reuse or contaminate test data."""
import pytest

from tabet import ledger as L

CSV = """division,season_from,season_to,family,role,status,quota,reference,note
E0,2018-19,2020-21,OLD,test,spent,1,x,closed
E0,2024-25,2025-26,core,development,spent,1,x,
E0,2008-09,2014-15,P,test,spent,3,x,closed set with quota 3
E0,2008-09,2014-15,Q,test,reserved,3,x,second family in the quota
SP1,2018-19,2020-21,G,test,reserved,1,x,
SP1,2014-15,2016-17,G,development,reserved,1,x,
SP1,2018-19,2020-21,Z,development,reserved,1,x,would contaminate G
E0,2018-19,2018-19,NEW,test,reserved,1,x,tries to reuse a spent test
"""


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "ledger.csv"
    p.write_text(CSV)
    return p


def test_season_range():
    assert L.season_range("2018-19", "2020-21") == ["2018-19", "2019-20", "2020-21"]
    assert L.season_range("1999-00", "2000-01") == ["1999-00", "2000-01"]


def test_a_registered_untouched_test_set_can_be_claimed(path):
    L.claim("G", "SP1", ["2014-15", "2015-16"], "development", path)
    with pytest.raises(L.LedgerError, match="not untouched"):   # Z registered a use of G's test seasons
        L.claim("G", "SP1", ["2018-19"], "test", path)


def test_unregistered_use_is_refused(path):
    with pytest.raises(L.LedgerError, match="has not registered"):
        L.claim("G", "SP1", ["2010-11"], "training", path)


def test_a_spent_test_set_is_never_untouched_again(path):
    with pytest.raises(L.LedgerError, match="not untouched"):
        L.claim("NEW", "E0", ["2018-19"], "test", path)


def test_a_closed_test_set_takes_families_up_to_its_quota(path):
    L.claim("Q", "E0", L.season_range("2008-09", "2014-15"), "test", path)   # second of three


def test_no_one_develops_on_a_reserved_test_set(path):
    with pytest.raises(L.LedgerError, match="reserved as a test set"):
        L.claim("Z", "SP1", ["2019-20"], "development", path)


def test_the_real_ledger_parses_and_is_consistent():
    led = L.load()
    assert set(led.role) <= L.ROLES and set(led.status) <= {"reserved", "spent"}
    assert not led.duplicated(["division", "season", "family", "role"]).any()
    # the live test is the only reserved test set until a new pre-registration adds one
    reserved_tests = led[(led.role == "test") & (led.status == "reserved")]
    assert set(reserved_tests.family) <= {"live", "G"}
