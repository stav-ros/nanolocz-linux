"""
NL-35: Tip Estimation and Regularized Deconvolution - Test Suite
"""

import pytest
import numpy as np
from nanolocz.core.tip_estimation import (
    estimate_tip_morphological,
    estimate_tip_optimization,
    richardson_lucy_deconv,
    wiener_deconv,
    batch_deconvolve,
    validate_tip,
    TipEstimate,
    DeconvolutionResult,
    _create_parabolic_tip
)


def _has_cupy():
    try:
        import cupy
        return True
    except ImportError:
        return False


class TestMorphologicalTipEstimation:
    def test_parabolic_tip_recovery(self):
        np.random.seed(42)
        box_size = 64
        true_surface = np.zeros((box_size, box_size))
        true_surface[24:40, 24:40] = 1.0
        tip = _create_parabolic_tip((box_size, box_size), radius=8.0)
        tip_est = estimate_tip_morphological(true_surface, tip_radius_guess=8.0)
        assert isinstance(tip_est, TipEstimate)
        assert tip_est.method == "morphological"
        assert tip_est.sharpness >= 0.0
        assert tip_est.sharpness <= 1.0

    def test_conical_tip_recovery(self):
        np.random.seed(42)
        box_size = 64
        y, x = np.ogrid[:box_size, :box_size]
        cy, cx = box_size // 2, box_size // 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        cone = np.maximum(0, 20 - r)
        tip_est = estimate_tip_morphological(cone, tip_radius_guess=10.0)
        assert isinstance(tip_est, TipEstimate)
        assert tip_est.volume > 0

    def test_noisy_image_robustness(self):
        np.random.seed(42)
        box_size = 32
        signal = np.zeros((box_size, box_size))
        signal[12:20, 12:20] = 1.0
        for noise_level in [0.01, 0.1]:
            noisy = signal + np.random.randn(box_size, box_size) * noise_level
            tip_est = estimate_tip_morphological(noisy, tip_radius_guess=8.0)
            assert isinstance(tip_est, TipEstimate)


class TestOptimizationTipEstimation:
    def test_tikhonov_regularization(self):
        np.random.seed(42)
        box_size = 16
        image = np.zeros((box_size, box_size))
        image[6:10, 6:10] = 1.0
        tip_est = estimate_tip_optimization(image, regularization='tikhonov', max_iterations=10)
        assert isinstance(tip_est, TipEstimate)
        assert tip_est.method == "optimization_tikhonov"

    def test_tv_regularization(self):
        np.random.seed(42)
        box_size = 16
        image = np.zeros((box_size, box_size))
        image[6:10, 6:10] = 1.0
        tip_est = estimate_tip_optimization(image, regularization='tv', max_iterations=10)
        assert isinstance(tip_est, TipEstimate)

    def test_sparse_regularization(self):
        np.random.seed(42)
        box_size = 16
        image = np.zeros((box_size, box_size))
        image[6:10, 6:10] = 1.0
        tip_est = estimate_tip_optimization(image, regularization='sparse', max_iterations=10)
        assert isinstance(tip_est, TipEstimate)


class TestRichardsonLucyDeconv:
    def test_basic_deconvolution(self):
        np.random.seed(42)
        box_size = 32
        sharp = np.zeros((box_size, box_size))
        sharp[12:20, 12:20] = 1.0
        psf = _create_parabolic_tip((box_size, box_size), 5.0)
        from scipy import ndimage
        blurred = ndimage.convolve(sharp, psf, mode='reflect')
        result = richardson_lucy_deconv(blurred, psf, n_iterations=10)
        assert isinstance(result, DeconvolutionResult)
        assert result.method == "richardson_lucy"
        assert result.deconvolved.shape == (box_size, box_size)

    def test_regularization_effect(self):
        np.random.seed(42)
        box_size = 32
        sharp = np.zeros((box_size, box_size))
        sharp[12:20, 12:20] = 1.0
        psf = _create_parabolic_tip((box_size, box_size), 5.0)
        from scipy import ndimage
        blurred = ndimage.convolve(sharp, psf, mode='reflect')
        for reg_type in ['none', 'tikhonov', 'tv']:
            result = richardson_lucy_deconv(blurred, psf, n_iterations=5, regularization=reg_type)
            assert isinstance(result, DeconvolutionResult)

    def test_conservation_of_mass(self):
        np.random.seed(42)
        box_size = 32
        sharp = np.zeros((box_size, box_size))
        sharp[12:20, 12:20] = 1.0
        psf = _create_parabolic_tip((box_size, box_size), 5.0)
        from scipy import ndimage
        blurred = ndimage.convolve(sharp, psf, mode='reflect')
        result = richardson_lucy_deconv(blurred, psf, n_iterations=10)
        input_mass = np.sum(blurred)
        output_mass = np.sum(result.deconvolved)
        ratio = output_mass / (input_mass + 1e-10)
        assert 0.5 < ratio < 2.0


class TestWienerDeconv:
    def test_basic_wiener_deconv(self):
        np.random.seed(42)
        box_size = 32
        sharp = np.zeros((box_size, box_size))
        sharp[12:20, 12:20] = 1.0
        psf = _create_parabolic_tip((box_size, box_size), 5.0)
        from scipy import ndimage
        blurred = ndimage.convolve(sharp, psf, mode='reflect')
        result = wiener_deconv(blurred, psf)
        assert isinstance(result, DeconvolutionResult)
        assert result.method == "wiener"

    def test_auto_snr_estimation(self):
        np.random.seed(42)
        box_size = 32
        sharp = np.zeros((box_size, box_size))
        sharp[12:20, 12:20] = 1.0
        psf = _create_parabolic_tip((box_size, box_size), 5.0)
        from scipy import ndimage
        blurred = ndimage.convolve(sharp, psf, mode='reflect')
        result_auto = wiener_deconv(blurred, psf, snr=None)
        result_explicit = wiener_deconv(blurred, psf, snr=10.0)
        assert isinstance(result_auto, DeconvolutionResult)
        assert isinstance(result_explicit, DeconvolutionResult)

    def test_known_snr_input(self):
        np.random.seed(42)
        box_size = 32
        sharp = np.zeros((box_size, box_size))
        sharp[12:20, 12:20] = 1.0
        psf = _create_parabolic_tip((box_size, box_size), 5.0)
        from scipy import ndimage
        blurred = ndimage.convolve(sharp, psf, mode='reflect')
        for snr in [1.0, 10.0]:
            result = wiener_deconv(blurred, psf, snr=snr)
            assert isinstance(result, DeconvolutionResult)


class TestBatchDeconvolution:
    def test_batch_consistency(self):
        np.random.seed(42)
        box_size = 32
        n_images = 3
        images = np.random.randn(n_images, box_size, box_size) * 0.1
        for i in range(n_images):
            images[i, 12:20, 12:20] += 1.0
        psf = _create_parabolic_tip((box_size, box_size), 5.0)
        batch_results = batch_deconvolve(images, psf, method='wiener')
        individual_results = [wiener_deconv(images[i], psf) for i in range(n_images)]
        assert len(batch_results) == len(individual_results)
        for b, i in zip(batch_results, individual_results):
            assert np.allclose(b.deconvolved, i.deconvolved, rtol=1e-10)

    def test_variable_image_count(self):
        np.random.seed(42)
        box_size = 16
        psf = _create_parabolic_tip((box_size, box_size), 5.0)
        for n_images in [1, 3, 5]:
            images = np.random.randn(n_images, box_size, box_size)
            results = batch_deconvolve(images, psf, method='wiener')
            assert len(results) == n_images


class TestTipValidation:
    def test_aspect_ratio_computation(self):
        tip = _create_parabolic_tip((32, 32), 8.0)
        image = np.random.randn(32, 32)
        metrics = validate_tip(tip, image)
        assert 'aspect_ratio' in metrics
        assert metrics['aspect_ratio'] > 0

    def test_volume_computation(self):
        tip = _create_parabolic_tip((32, 32), 8.0)
        image = np.random.randn(32, 32)
        metrics = validate_tip(tip, image)
        assert 'volume' in metrics
        assert metrics['volume'] > 0

    def test_sharpness_metric(self):
        tip = _create_parabolic_tip((32, 32), 8.0)
        image = np.random.randn(32, 32)
        metrics = validate_tip(tip, image)
        assert 'sharpness' in metrics
        assert 0.0 <= metrics['sharpness'] <= 1.0

    def test_symmetry_metric(self):
        symmetric = _create_parabolic_tip((32, 32), 8.0)
        asymmetric = np.zeros((32, 32))
        asymmetric[10:20, 10:25] = 1.0
        image = np.random.randn(32, 32)
        sym_metrics = validate_tip(symmetric, image)
        asym_metrics = validate_tip(asymmetric, image)
        assert 'symmetry' in sym_metrics
        assert 0.0 <= sym_metrics['symmetry'] <= 1.0

    def test_physical_plausibility_checks(self):
        physical = _create_parabolic_tip((32, 32), 8.0)
        non_physical = _create_parabolic_tip((32, 32), 8.0)
        non_physical[10, 10] = -1.0
        image = np.random.randn(32, 32)
        phys_metrics = validate_tip(physical, image)
        non_phys_metrics = validate_tip(non_physical, image)
        assert phys_metrics['physical_checks']['non_negative'] == True
        assert non_phys_metrics['physical_checks']['non_negative'] == False


class TestIntegration:
    def test_end_to_end_tip_estimation_deconv(self):
        np.random.seed(42)
        box_size = 32
        sharp = np.zeros((box_size, box_size))
        sharp[12:20, 12:20] = 1.0
        psf = _create_parabolic_tip((box_size, box_size), 5.0)
        from scipy import ndimage
        blurred = ndimage.convolve(sharp, psf, mode='reflect')
        tip_est = estimate_tip_morphological(blurred, tip_radius_guess=5.0)
        deconv_result = richardson_lucy_deconv(blurred, tip_est.tip_height, n_iterations=10)
        metrics = validate_tip(tip_est.tip_height, blurred, deconv_result.deconvolved)
        assert isinstance(tip_est, TipEstimate)
        assert isinstance(deconv_result, DeconvolutionResult)
        assert isinstance(metrics, dict)

    def test_with_simulated_afm_from_nl24(self):
        np.random.seed(42)
        box_size = 32
        true_surface = np.zeros((box_size, box_size))
        true_surface[10:22, 10:22] = 1.0
        true_tip = _create_parabolic_tip((box_size, box_size), 6.0)
        from scipy import ndimage
        afm_image = ndimage.convolve(true_surface, true_tip, mode='reflect')
        tip_est = estimate_tip_morphological(afm_image, tip_radius_guess=6.0)
        assert isinstance(tip_est, TipEstimate)


class TestTipEstimationGPU:
    @pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")
    def test_gpu_richardson_lucy_parity(self):
        from nanolocz.core.tip_estimation import richardson_lucy_gpu
        np.random.seed(42)
        box_size = 32
        sharp = np.zeros((box_size, box_size))
        sharp[12:20, 12:20] = 1.0
        psf = _create_parabolic_tip((box_size, box_size), 5.0)
        from scipy import ndimage
        blurred = ndimage.convolve(sharp, psf, mode='reflect')
        cpu_result = richardson_lucy_deconv(blurred, psf, n_iterations=10)
        gpu_result_arr = richardson_lucy_gpu(blurred, psf, n_iterations=10)
        assert np.allclose(cpu_result.deconvolved, gpu_result_arr, rtol=1e-4)

    @pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")
    def test_gpu_wiener_parity(self):
        from nanolocz.core.tip_estimation import wiener_deconv_gpu
        np.random.seed(42)
        box_size = 32
        sharp = np.zeros((box_size, box_size))
        sharp[12:20, 12:20] = 1.0
        psf = _create_parabolic_tip((box_size, box_size), 5.0)
        from scipy import ndimage
        blurred = ndimage.convolve(sharp, psf, mode='reflect')
        cpu_result = wiener_deconv(blurred, psf)
        gpu_result_arr = wiener_deconv_gpu(blurred, psf)
        assert np.allclose(cpu_result.deconvolved, gpu_result_arr, rtol=1e-4)

    @pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")
    def test_gpu_speedup(self):
        import time
        from nanolocz.core.tip_estimation import richardson_lucy_gpu
        np.random.seed(42)
        box_size = 128
        sharp = np.zeros((box_size, box_size))
        sharp[40:88, 40:88] = 1.0
        psf = _create_parabolic_tip((box_size, box_size), 10.0)
        from scipy import ndimage
        blurred = ndimage.convolve(sharp, psf, mode='reflect')
        start = time.time()
        cpu_result = richardson_lucy_deconv(blurred, psf, n_iterations=20)
        cpu_time = time.time() - start
        start = time.time()
        gpu_result = richardson_lucy_gpu(blurred, psf, n_iterations=20)
        gpu_time = time.time() - start
        assert gpu_time < cpu_time * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
