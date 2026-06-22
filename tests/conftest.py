import sys
import types


def _install_flux_stub() -> None:
    flux_module = types.ModuleType("flux")
    flux_job_module = types.ModuleType("flux.job")
    flux_job_executor_module = types.ModuleType("flux.job.executor")
    flux_resource_module = types.ModuleType("flux.resource")
    flux_resource_list_module = types.ModuleType("flux.resource.list")

    class _DummyExecutor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, *args, **kwargs):
            return _DummyFuture(*args, **kwargs)

    class _DummyFuture:
        def __init__(self, fn=None, *args, **kwargs):
            self._exception = None
            self._result = None
            if fn is not None:
                try:
                    self._result = fn(*args, **kwargs)
                except Exception as exc:
                    self._exception = exc

        def result(self, timeout=None):
            if self._exception is not None:
                raise self._exception
            return self._result

        def exception(self, timeout=None):
            return self._exception

    class _DummyFlux:
        def rpc(self, *_args, **_kwargs):
            class _RpcResult:
                def get(self):
                    return None

            return _RpcResult()

    class _DummyResources:
        class free:
            ranks = [1]
            ncores = 1
            ngpus = 0

    class _ResourceList:
        def get(self):
            return _DummyResources()

    flux_module.Flux = _DummyFlux
    flux_job_module.FluxExecutor = _DummyExecutor
    flux_job_module.FluxExecutorFuture = _DummyFuture
    flux_resource_list_module.resource_list = lambda _handle: _ResourceList()
    flux_resource_module.list = flux_resource_list_module
    flux_module.resource = flux_resource_module
    flux_module.job = flux_job_module
    flux_job_module.executor = flux_job_executor_module

    sys.modules["flux"] = flux_module
    sys.modules["flux.job"] = flux_job_module
    sys.modules["flux.job.executor"] = flux_job_executor_module
    sys.modules["flux.resource"] = flux_resource_module
    sys.modules["flux.resource.list"] = flux_resource_list_module


try:
    import flux  # noqa: F401
except ModuleNotFoundError:
    _install_flux_stub()


def _install_redis_stub() -> None:
    redis_module = types.ModuleType("redis")

    class _PlaceholderRedis:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("redis.Redis stub should be monkeypatched in tests")

    redis_module.Redis = _PlaceholderRedis
    sys.modules["redis"] = redis_module


try:
    import redis  # noqa: F401
except ModuleNotFoundError:
    _install_redis_stub()
