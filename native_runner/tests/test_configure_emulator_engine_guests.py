from native_runner.training.v4 import configure_emulator_engine_guests as guests


def test_merge_activity_manager_constants_keeps_existing_keys():
    merged = guests.merge_activity_manager_constants(
        "max_cached_processes=32,power_check_max_cpu_4=2", {"power_check_max_cpu_4": "100", "power_check_interval": "9"}
    )
    assert merged == "max_cached_processes=32,power_check_max_cpu_4=100,power_check_interval=9"


def test_merge_activity_manager_constants_from_unset():
    assert guests.merge_activity_manager_constants("", {"a": "1"}) == "a=1"


def test_relax_power_check_writes_and_verifies(monkeypatch):
    stored = {"value": "null"}
    calls = []

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_adb(serial, *args, **kwargs):
        calls.append(args)
        if args[:4] == ("shell", "settings", "get", "global"):
            return Result(stored["value"] + "\n")
        if args[:4] == ("shell", "settings", "put", "global"):
            stored["value"] = args[5].strip("'")
        return Result("")

    monkeypatch.setattr(guests, "adb", fake_adb)
    guests.relax_power_check("guest")
    assert "power_check_max_cpu_4=100" in stored["value"]
    assert "power_check_interval=86400000" in stored["value"]
