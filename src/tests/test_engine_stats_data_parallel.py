from vllm_router.stats.engine_stats import EngineStats

TWO_ENGINES = """# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 5.0
vllm:num_requests_running{engine="1",model_name="m"} 1.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="m"} 2.0
vllm:num_requests_waiting{engine="1",model_name="m"} 7.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.95
vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.10
# HELP vllm:gpu_prefix_cache_hits Prefix cache hits, in terms of number of cached tokens.
# TYPE vllm:gpu_prefix_cache_hits counter
vllm:gpu_prefix_cache_hits_total{engine="0",model_name="m"} 100.0
vllm:gpu_prefix_cache_hits_total{engine="1",model_name="m"} 40.0
# HELP vllm:gpu_prefix_cache_queries Prefix cache queries, in terms of number of queried tokens.
# TYPE vllm:gpu_prefix_cache_queries counter
vllm:gpu_prefix_cache_queries_total{engine="0",model_name="m"} 300.0
vllm:gpu_prefix_cache_queries_total{engine="1",model_name="m"} 100.0
"""


def test_request_counts_are_summed_across_engines():
    stats = EngineStats.from_vllm_scrape(TWO_ENGINES)
    assert stats.num_running_requests == 6
    assert stats.num_queuing_requests == 9


def test_kv_cache_usage_is_the_most_loaded_engine():
    # A mean (0.525) would hide that engine 0 is nearly full.
    stats = EngineStats.from_vllm_scrape(TWO_ENGINES)
    assert stats.gpu_cache_usage_perc == 0.95


def test_prefix_cache_counters_are_summed_across_engines():
    stats = EngineStats.from_vllm_scrape(TWO_ENGINES)
    assert stats.gpu_prefix_cache_hits_total == 140
    assert stats.gpu_prefix_cache_queries_total == 400


def _swap_engine_order(text):
    lines = text.splitlines()
    for i in range(len(lines) - 1):
        if 'engine="0"' in lines[i] and 'engine="1"' in lines[i + 1]:
            lines[i], lines[i + 1] = lines[i + 1], lines[i]
    return "\n".join(lines) + "\n"


def test_engine_order_does_not_matter():
    # Before the fix, whichever engine was parsed last won.
    assert EngineStats.from_vllm_scrape(TWO_ENGINES) == EngineStats.from_vllm_scrape(
        _swap_engine_order(TWO_ENGINES)
    )
