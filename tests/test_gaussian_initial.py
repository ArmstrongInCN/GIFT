"""The initial distribution and global split must not depend on model results."""
import numpy as np
import pytest

from gift.gaussian_initial import ENSEMBLE_RMS, initial_hat, specification
from gift.prediction_cohorts import GAUSSIAN, partitions


def test_appended_disjoint_ids():
    original, gaussian = partitions(), partitions(GAUSSIAN)
    a = set(original['training'] + original['validation'] + original['test'])
    groups = [set(gaussian[k]) for k in ('training','validation','test')]
    assert [len(x) for x in groups] == [1000,40,180]
    assert set.union(*groups) == set(range(1220,2440))
    assert all(not a & x for x in groups)
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))


def test_draws_independent_of_batch_and_order():
    together = initial_hat([1220,1221])
    assert np.array_equal(together[0],initial_hat([1220])[0])
    assert np.array_equal(together[::-1],initial_hat([1221,1220]))
    assert not np.array_equal(together[0],together[1])
    assert together.dtype == np.complex64
    assert np.array_equal(together[:,0,0],np.zeros(2))
    # Inspect the stored coefficients in double precision, independently of
    # the float32 inverse-FFT rounding error of the production solver.
    fields = np.fft.ifft2(together.astype(np.complex128))
    assert np.abs(fields.imag).max() < 1e-12


def test_ensemble_amplitude_not_per_sample_renormalized():
    fields = np.fft.ifft2(initial_hat(list(range(1220,1476)))).real
    rms = np.sqrt(np.mean(fields**2,axis=(1,2)))
    assert abs(np.sqrt(np.mean(fields**2))/ENSEMBLE_RMS-1) < .03
    assert rms.std() > .025
    assert specification()['samplewise_rms_normalization'] is False


@pytest.mark.parametrize('ids', [[],[1220,1220],[-1],[1.5],[True]])
def test_invalid_ids(ids):
    with pytest.raises(ValueError):
        initial_hat(ids)
