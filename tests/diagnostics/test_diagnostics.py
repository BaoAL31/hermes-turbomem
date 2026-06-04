from hermes_turbomem.diagnostics import IndexLogger, IndexMetrics


class TestIndexLogger:
    def test_log_and_retrieve(self) -> None:
        log = IndexLogger(max_entries=100)
        log.log("index", "INFO", "hello")
        log.log("embed", "WARN", "warning")
        entries = log.get_logs()
        assert len(entries) == 2
        assert entries[0].message == "hello"
        assert entries[1].message == "warning"

    def test_filter_by_category(self) -> None:
        log = IndexLogger(max_entries=100)
        log.log("index", "INFO", "index msg")
        log.log("embed", "INFO", "embed msg")
        log.log("parse", "WARN", "parse warn")
        entries = log.get_logs(category="embed")
        assert len(entries) == 1
        assert entries[0].message == "embed msg"

    def test_filter_by_level(self) -> None:
        log = IndexLogger(max_entries=100)
        log.log("index", "INFO", "info")
        log.log("embed", "WARN", "warn")
        log.log("parse", "ERROR", "error")
        entries = log.get_logs(level="WARN")
        assert len(entries) == 1
        assert entries[0].message == "warn"

    def test_filter_by_category_and_level(self) -> None:
        log = IndexLogger(max_entries=100)
        log.log("parse", "WARN", "parse warn")
        log.log("parse", "ERROR", "parse error")
        log.log("embed", "WARN", "embed warn")
        entries = log.get_logs(category="parse", level="WARN")
        assert len(entries) == 1
        assert entries[0].message == "parse warn"

    def test_limit(self) -> None:
        log = IndexLogger(max_entries=100)
        for i in range(10):
            log.log("index", "INFO", f"msg {i}")
        entries = log.get_logs(limit=3)
        assert len(entries) == 3
        assert entries[-1].message == "msg 9"
        assert entries[0].message == "msg 7"

    def test_empty_returns_no_entries(self) -> None:
        log = IndexLogger(max_entries=100)
        assert log.get_logs() == []

    def test_ring_buffer_max_entries(self) -> None:
        log = IndexLogger(max_entries=3)
        for i in range(5):
            log.log("index", "INFO", f"msg {i}")
        entries = log.get_logs()
        assert len(entries) == 3
        assert entries[0].message == "msg 2"
        assert entries[-1].message == "msg 4"


class TestIndexMetrics:
    def test_increment(self) -> None:
        m = IndexMetrics()
        m.increment("embed_call")
        m.increment("embed_call")
        snap = m.snapshot()
        assert snap["embed_call_count"] == 2

    def test_record_timing_index(self) -> None:
        m = IndexMetrics()
        m.record_timing("index", 150.0)
        snap = m.snapshot()
        assert snap["index_run_count"] == 1
        assert snap["total_index_duration_ms"] == 150.0

    def test_record_timing_search(self) -> None:
        m = IndexMetrics()
        m.record_timing("search", 42.5)
        snap = m.snapshot()
        assert snap["search_call_count"] == 1
        assert snap["total_search_duration_ms"] == 42.5

    def test_increment_parse_error(self) -> None:
        m = IndexMetrics()
        m.increment("parse_error")
        m.increment("parse_error")
        snap = m.snapshot()
        assert snap["parse_error_count"] == 2

    def test_increment_embed_error(self) -> None:
        m = IndexMetrics()
        m.increment("embed_error")
        snap = m.snapshot()
        assert snap["embed_error_count"] == 1

    def test_snapshot_includes_all_keys(self) -> None:
        m = IndexMetrics()
        snap = m.snapshot()
        expected_keys = {
            "embed_call_count",
            "index_run_count",
            "search_call_count",
            "parse_error_count",
            "embed_error_count",
            "total_index_duration_ms",
            "total_search_duration_ms",
        }
        assert set(snap.keys()) == expected_keys

    def test_accumulate_per_process(self) -> None:
        m = IndexMetrics()
        m.increment("embed_call")
        m.record_timing("index", 100.0)
        m.record_timing("search", 50.0)
        snap1 = m.snapshot()
        assert snap1["embed_call_count"] == 1
        assert snap1["total_index_duration_ms"] == 100.0
        assert snap1["total_search_duration_ms"] == 50.0
        m.increment("embed_call")
        m.record_timing("index", 200.0)
        snap2 = m.snapshot()
        assert snap2["embed_call_count"] == 2
        assert snap2["total_index_duration_ms"] == 300.0  # accumulated
        assert snap2["total_search_duration_ms"] == 50.0  # unchanged
