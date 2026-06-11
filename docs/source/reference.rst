.. _user-reference:

Configuration and behavior reference
====================================

This page lists **every user-visible switch** and the **artifacts** MatEnsemble writes, so you can
debug a run without spelunking the source.

:class:`~matensemble.pipeline.Pipeline` constructor
----------------------------------------------------

``basedir`` (optional :class:`str`)
    If omitted, the workflow root directory is created under :meth:`pathlib.Path.cwd`. If provided,
    the stamped ``matensemble_workflow-…`` directory is created **inside** this path. The parent of
    the stamped directory is what gets added to ``PYTHONPATH`` for Python chores (see :doc:`design`).

``Pipeline.chore`` decorator factory
------------------------------------

Each optional argument becomes part of :class:`~matensemble.model.Resources` and affects the Flux jobspec.

``name``
    If set, chore IDs look like ``chore-<name>-NNNN``; otherwise ``chore-<funcname>-NNNN``.

``num_tasks``, ``cores_per_task``, ``gpus_per_task``
    Passed directly to :meth:`flux.job.JobspecV1.from_command`. All must satisfy the validations in
    :class:`~matensemble.model.Resources` (for example ``num_tasks >= 1``).

``mpi``
    When true, MatEnsemble sets the Flux shell option ``mpi=pmi2`` on the chorespec. Your MPI launcher and
    site modules must match what Flux expects.

``env``
    Extra environment variables merged on top of the base environment. For Python chores, MatEnsemble also
    injects or prepends ``PYTHONPATH`` pointing at the source root (parent of the workflow directory).

``inherit_env``
    If true (the default in code), the chorespec starts from ``os.environ`` in the **submitting** process and
    applies ``env`` overrides; if false, only the keys in ``env`` are sent (still including MatEnsemble’s
    ``PYTHONPATH`` tweaks for Python chores).

**Nesting restriction:** decorated callables must be **module-level** functions (or other importable
qualified names). Closures and nested ``def``\ s raise :exc:`ValueError` because ``"<locals>"`` appears
in the qualified name.

``Pipeline.exec``
-----------------

Creates a :class:`~matensemble.chore.Chore` with :attr:`~matensemble.model.ChoreType.EXECUTABLE`. The command
may be a string (split with :mod:`shlex`) or a pre-split argv list. No automatic ``PYTHONPATH`` injection
occurs unless you pass it through ``env``. There is **no dependency tracking** for executable chores; use
a Python chore if you need DAG edges.

``Pipeline.submit``
-------------------

``write_restart_freq`` (:class:`int` or ``None``; default ``None``)
    Restart/checkpoint files are not supported yet. Leave this as ``None``. Passing an integer raises
    :exc:`NotImplementedError` before work is submitted.

``buffer_time`` (:class:`float`; default ``1.0``)
    Passed to :func:`concurrent.futures.wait` as the ``timeout`` when draining Flux futures; also used as a
    :func:`time.sleep` after each individual submission. Set to ``0.0`` for minimal spacing.

``log_delay`` (:class:`float`; default ``5.0``)
    The amount of time the logging thread will wait before updating the logs. Dafaults to every ``5.0`` seconds.

``set_cpu_affinity`` / ``set_gpu_affinity`` (default ``True`` / ``False``)
    Control Flux shell options ``cpu-affinity`` and ``gpu-affinity`` (GPU option only applies when the chore
    requests GPUs).

``adaptive`` (default ``True``)
    If true (and no custom ``processing_strategy`` is given), use :class:`~matensemble.strategy.AdaptiveStrategy`
    so newly ready chores can be submitted inside the completion loop. If false, :class:`~matensemble.strategy.NonAdaptiveStrategy`
    only drains futures.

``dynopro``
    Reserved for whole-node dynopro workflows. General users should leave this ``False``.

    .. warning::

       The in-tree ``dynopro`` stack is experimental and is not part of the stable public API.

``processing_strategy``
    Supply your own :class:`~matensemble.strategy.FutureProcessingStrategy` to replace adaptive / non-adaptive
    selection entirely.

``dashboard`` (default ``False``)
    When true, starts a Starlette/uvicorn server on ``0.0.0.0:8000`` serving the packaged static UI
    and JSON status (see :func:`matensemble.utils.setup_dashboard`).

``status.json`` schema
----------------------

Written atomically (temp file + rename) by :class:`~matensemble.logger.StatusWriter`. Keys:

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Key
     - Meaning
   * - ``nodes``
     - Free Flux node count after draining broker rank 0.
   * - ``coresPerNode``
     - ``total_free_cores // nodes`` from Flux resource RPCs.
   * - ``gpusPerNode``
     - ``total_free_gpus // nodes`` (integer division).
   * - ``pending``
     - Chores waiting in ready + blocked queues (sum of not-yet-finished backlog).
   * - ``running``
     - Chores with active Flux futures.
   * - ``completed``
     - Successful chore IDs recorded in order.
   * - ``failed``
     - Count of failures recorded in the manager.
   * - ``freeCores`` / ``freeGpus``
     - Current free resources as reported by Flux at the last loop iteration.

The dashboard’s ``GET /api/status`` returns the same object, or zeros if the file is missing.

Per-chore artifacts
-------------------

``stdout`` / ``stderr``
    Standard streams from Flux. MatEnsemble appends human-readable blocks to ``stderr`` when futures raise
    Python exceptions or return non-zero shell exit codes.

``metadata.json``
    Debug snapshot of id, chore_type, argv, resource struct, function import path, dependency IDs, and serialized
    arguments

``chore.pickle``
    Pickle written at submit time; the worker reloads this file.

``result.pickle``
    Python return value. Downstream chores load ``../<dep_chore_id>/result.pickle`` via
    :func:`matensemble.runtime_worker._load_dep_result`.

Failure ``reason`` strings (internal)
---------------------------------------

Recorded in :meth:`~matensemble.manager.FluxManager._record_failure` entries:

* ``chore_exceeds_allocation`` — resources larger than the Flux allocation; chore is skipped and dependents fail.
* ``submit_exception`` — :meth:`~matensemble.fluxlet.Fluxlet.submit` raised before a future was registered.
* ``exception`` — future completion raised (wrapper or process error surfaced as an exception).
* ``nonzero_exit:<rc>`` — future returned a non-zero integer exit code.
* ``dependency_failed`` — cascaded skip because an upstream chore failed.
