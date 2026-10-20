"""Training-state compatibility independent of provenance and IO settings."""
import os
from pathlib import Path


def output_path(value, project_root, *read_only_roots):
    """Resolve a run output while protecting project code, published products and read-only roots.

    read_only_roots and configured data/upstream roots must not contain the
    output. This function checks placement only; creating a fresh directory
    or opening an existing resume journal remains the caller's responsibility.
    """
    output, root = Path(value).resolve(), Path(project_root).resolve()
    protected = [root / name for name in ("src", "training", "adapters", "experiments", "scripts", "artifacts", "results")]
    protected.extend(Path(value).resolve() for value in read_only_roots if value is not None)
    protected.extend(Path(os.environ[name]).resolve() for name in ("GIFT_DATA_ROOT", "GIFT_EXTERNAL_ROOT") if os.environ.get(name))
    if output == root or any(output == path or path in output.parents for path in protected):
        raise ValueError("Run output must not overwrite code, data or published results")
    return output

def resume_configuration(value):
    """Return recursively filtered configuration used for semantic resume comparison.

    Keep counts, seeds, shapes, loss/optimizer choices and input selections.
    Drop descriptive paths/source/runtime fields and operational IO settings.
    Already-loaded scientific fingerprints are compared separately by
    validate_resume so legacy records can omit them without a second file read.
    """
    metadata = {
        "sources", "source_identity", "source_sha256", "external_sources", "upstream",
        "runtime", "numerical_profile", "device", "gpu", "execution",
        "execution_requested", "execution_reason", "cache_observations", "checkpoint_interval",
        "provenance", "data_profile", "profile", "file", "data_file", "bytes", "dataset_bytes", "data_bytes",
        "sha256", "file_sha256", "observations_sha256", "sha256_verified",
        "full_input_checks_performed", "published_dataset_identity", "numerical_reproduction_verified",
        "generator_sha256", "manifest_sha256", "continuation", "directory", "path",
        "nadam_checkpoint_interval", "time_grid_recipe", "byte_order", "declared_backend", "cohort",
        "h5py", "torch_threads", "torch_interop_threads", "omp_threads", "mkl_threads",
        "consumed_files", "generation_attempts",
    }
    if isinstance(value, dict):
        return {key: resume_configuration(item) for key, item in value.items()
                if key not in metadata and not key.endswith("_sha256")}
    if isinstance(value, (list, tuple)):
        return [resume_configuration(item) for item in value]
    return value


def validate_resume(identity, saved):
    """Require compatible training semantics and already-loaded scientific inputs.

    Ignore paths, source records and execution settings in the recursive
    configuration comparison. Generator-only and branch-only fields are
    compared by role. Where both journals contain decoded-array or weight
    fingerprints, those values must agree; older journals may omit them.
    """
    current, previous = resume_configuration(identity), resume_configuration(saved)
    irrelevant = {
        "gift_prediction_generator": ("branch_phases", "rollout_batch_size", "fixture_only"),
        "gift_prediction_branch": ("generator_phases", "generator_learning_rates", "generator_seed", "rank", "fixture_only"),
    }.get(current.get("role"), ())
    for value in (current, previous):
        for key in irrelevant:
            value.get("configuration", {}).pop(key, None)
    if current != previous:
        raise ValueError("Resume training configuration differs")
    # These fingerprints come from arrays/weights that the consumer already
    # loaded. Older journals may lack them; do not re-read files to manufacture
    # a new identity for an old checkpoint.
    scientific = {"observations_sha256", "generator_weights_sha256", "selected_arrays_sha256"}
    def compare(current, previous):
        """Walk common dictionary fields to compare available scientific fingerprints.

        A missing or None legacy fingerprint supplies no evidence of a mismatch;
        this walk deliberately does not read or hash an input file.
        """
        if not isinstance(current, dict) or not isinstance(previous, dict):
            return
        for key in current.keys() & previous.keys():
            if key in scientific and current[key] and previous[key] and current[key] != previous[key]:
                raise ValueError("Resume scientific inputs differ")
            compare(current[key], previous[key])
    compare(identity, saved)


def validate_training_state(payload):
    """Validate explicit progress counters before applying model/optimizer state.

    An epoch, when present, is a nonnegative integer and has one history row
    per committed epoch. Consecutive global epoch/update checks apply only
    when rows and payload declare that global cursor; phase-local histories
    are not reinterpreted as a second global epoch sequence.
    """
    epoch = payload.get("completed_epochs", payload.get("epoch"))
    if epoch is not None and (type(epoch) is not int or epoch < 0):
        raise ValueError("Invalid checkpoint epoch")
    updates = payload.get("optimizer_updates")
    if updates is not None and (type(updates) is not int or updates < 0):
        raise ValueError("Invalid checkpoint optimizer update count")
    history = payload.get("history")
    if history is not None and epoch is not None and len(history) != epoch:
        raise ValueError("Checkpoint history and epoch disagree")
    # Phase-based controllers may restart their row-level epoch at each phase.
    # Only journals with an explicit global epoch use a global history cursor.
    if history and epoch is not None and all(isinstance(row, dict) and "epoch" in row for row in history):
        if any(row["epoch"] != index + 1 for index, row in enumerate(history)):
            raise ValueError("Checkpoint epoch history is not consecutive")
        recorded = history[-1].get("cumulative_optimizer_updates", history[-1].get("optimizer_updates"))
        if updates is not None and recorded is not None and recorded != updates:
            raise ValueError("Checkpoint history and optimizer updates disagree")


