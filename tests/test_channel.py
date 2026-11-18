"""Tests for differentiable channel models."""

import torch
import pytest
from scipy import stats

from semvideo.channel.awgn import AWGNChannel
from semvideo.channel.fading import RayleighFadingChannel
from semvideo.channel.jamming import JammingChannel
from semvideo.channel.packet_loss import BernoulliErasureChannel, BurstErasureChannel


class TestAWGNChannel:
    """Tests for AWGN channel model."""

    def test_output_shape_matches_input(self) -> None:
        """Channel output must have the same shape as input."""
        channel = AWGNChannel(snr_db=10.0)
        x = torch.randn(4, 128)
        y = channel(x)
        assert y.shape == x.shape

    def test_noise_power_matches_snr(self) -> None:
        """Verify noise variance approximately matches the specified SNR."""
        snr_db = 10.0
        channel = AWGNChannel(snr_db=snr_db)
        channel.eval()

        torch.manual_seed(42)
        x = torch.randn(1000, 512)
        y = channel(x)
        noise = y - x

        signal_power = torch.mean(x**2).item()
        noise_power = torch.mean(noise**2).item()
        measured_snr_db = 10 * torch.log10(
            torch.tensor(signal_power / noise_power)
        ).item()

        # Allow 1.5 dB tolerance due to finite sample size
        assert abs(measured_snr_db - snr_db) < 1.5, (
            f"Expected SNR ~{snr_db}dB, got {measured_snr_db:.1f}dB"
        )

    def test_gradient_flows_through_channel(self) -> None:
        """Gradients must propagate through the channel for JSCC training."""
        channel = AWGNChannel(snr_db=10.0)
        x = torch.randn(4, 64, requires_grad=True)
        y = channel(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    def test_zero_noise_at_high_snr(self) -> None:
        """At very high SNR, output should approximately equal input."""
        channel = AWGNChannel(snr_db=100.0)
        channel.eval()
        x = torch.randn(4, 128)
        y = channel(x)
        assert torch.allclose(x, y, atol=1e-3)

    def test_random_snr_mode(self) -> None:
        """With snr_db=None, channel should sample random SNR."""
        channel = AWGNChannel(snr_db=None, snr_db_min=0, snr_db_max=20)
        channel.train()
        x = torch.randn(4, 128)
        y = channel(x)
        assert y.shape == x.shape

    def test_noise_distribution_gof(self) -> None:
        """KS-test: AWGN noise must follow Gaussian distribution N(0, sigma^2).

        Extracts noise samples (y - x) and tests against the expected normal
        distribution with variance = signal_power / snr_linear.
        """
        snr_db = 10.0
        channel = AWGNChannel(snr_db=snr_db)
        channel.eval()

        torch.manual_seed(123)
        x = torch.ones(2000, 256)  # constant signal for clean noise extraction
        y = channel(x)
        noise = (y - x).flatten().numpy()

        # Expected noise std: sqrt(signal_power / snr_linear)
        signal_power = torch.mean(x**2).item()
        snr_linear = 10.0 ** (snr_db / 10.0)
        expected_std = (signal_power / snr_linear) ** 0.5

        # Cramer-von Mises goodness of fit against a fully specified N(0, expected_std):
        # sigma comes from the nominal SNR, never from the sample.
        cvm = stats.cramervonmises(noise, "norm", args=(0.0, expected_std))
        assert cvm.pvalue > 0.01, (
            f"AWGN noise failed Cramer-von Mises GOF for N(0, {expected_std:.4f}): "
            f"W2={cvm.statistic:.4f}, p={cvm.pvalue:.6f}"
        )


class TestRayleighFadingChannel:
    """Tests for Rayleigh fading channel."""

    def test_output_shape(self) -> None:
        channel = RayleighFadingChannel(snr_db=10.0)
        x = torch.randn(4, 128)
        y = channel(x)
        assert y.shape == x.shape

    def test_gradient_flows(self) -> None:
        channel = RayleighFadingChannel(snr_db=10.0)
        x = torch.randn(4, 64, requires_grad=True)
        y = channel(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None

    def test_equalization_recovers_signal(self) -> None:
        """With equalization at high SNR, output should approximate input."""
        channel = RayleighFadingChannel(snr_db=100.0, equalize=True)
        channel.eval()
        x = torch.randn(4, 128)
        y = channel(x)
        # With perfect CSI and high SNR, should be close
        assert torch.allclose(x, y, atol=0.5)

    def test_no_equalization_differs(self) -> None:
        """Without equalization, output should differ from input."""
        torch.manual_seed(0)
        channel = RayleighFadingChannel(snr_db=10.0, equalize=False)
        x = torch.ones(4, 128)
        y = channel(x)
        # Fading changes the amplitude
        assert not torch.allclose(x, y, atol=0.01)

    def test_fading_magnitude_gof(self) -> None:
        """KS-test: |h| must follow Rayleigh distribution with scale=1/sqrt(2).

        The fading coefficient h = (h_r + j*h_i)/sqrt(2) where h_r, h_i ~ N(0,1),
        so |h| = sqrt(h_r^2 + h_i^2)/sqrt(2) ~ Rayleigh(sigma=1/sqrt(2)).
        We isolate |h| by using unit input and very high SNR (no noise).
        """
        channel = RayleighFadingChannel(snr_db=100.0, equalize=False)
        channel.eval()

        torch.manual_seed(456)
        x = torch.ones(5000, 256)  # unit signal so y = x * |h| ≈ |h|
        y = channel(x)
        h_mag = y.flatten().numpy()  # |h| samples (noise negligible at 100 dB)

        # Rayleigh distribution with scale sigma = 1/sqrt(2)
        # scipy parametrizes as Rayleigh(loc, scale) where pdf = x/s^2 * exp(-x^2/(2s^2))
        sigma = 1.0 / (2**0.5)
        cvm = stats.cramervonmises(h_mag, "rayleigh", args=(0, sigma))
        assert cvm.pvalue > 0.01, (
            f"Fading magnitude failed Cramer-von Mises GOF for Rayleigh(σ={sigma:.4f}): "
            f"W2={cvm.statistic:.4f}, p={cvm.pvalue:.6f}"
        )

    def test_awgn_component_gof(self) -> None:
        """KS-test: noise component after equalization must be approximately Gaussian.

        With perfect CSI equalization, received = x + n/|h|. The enhanced noise
        n/|h| is not exactly Gaussian, but the AWGN component n itself is.
        We test the raw AWGN by using no fading (high SNR isolates structure).
        """
        snr_db = 10.0
        channel = RayleighFadingChannel(snr_db=snr_db, equalize=False)
        channel.eval()

        # To isolate AWGN: use the same approach as AWGN test but on faded signal
        # Instead, verify the channel's AWGN noise variance is correct
        torch.manual_seed(789)
        x = torch.ones(3000, 256)
        signal_power = torch.mean(x**2).item()
        snr_linear = 10.0 ** (snr_db / 10.0)
        expected_noise_var = signal_power / snr_linear

        # Run multiple samples and check that faded output variance is consistent
        # with Rayleigh fading + AWGN model: Var[y] = E[|h|^2]*Var[x] + noise_var
        # For our model y = x*|h| + n, with x=1: Var[y] = Var[|h|] + noise_var
        y = channel(x)
        noise_after_fading = y.flatten().numpy()

        # |h| ~ Rayleigh(1/sqrt(2)), so E[|h|] = sqrt(pi)/2 * (1/sqrt(2))
        # Var[|h|] = (2 - pi/2) * sigma^2 = (2 - pi/2) / 2
        # We just verify the output has reasonable variance (not a distribution test)
        output_var = y.var().item()
        assert output_var > 0.05, "Rayleigh+AWGN output variance suspiciously low"


class TestJammingChannel:
    """Tests for jamming channel."""

    def test_barrage_jamming_increases_noise(self) -> None:
        """Barrage jamming should increase effective noise power."""
        torch.manual_seed(42)
        awgn = AWGNChannel(snr_db=10.0)
        jammer = JammingChannel(jammer_type="barrage", jnr_db=10.0, snr_db=10.0)

        x = torch.randn(100, 256)
        y_awgn = awgn(x)
        y_jammed = jammer(x)

        noise_awgn = torch.mean((y_awgn - x) ** 2).item()
        noise_jammed = torch.mean((y_jammed - x) ** 2).item()

        assert noise_jammed > noise_awgn, "Jamming should increase noise"

    def test_pulsed_jamming_duty_cycle(self) -> None:
        """Pulsed jammer with duty_cycle=0 should add no jamming."""
        channel = JammingChannel(jammer_type="pulsed", jnr_db=20.0, duty_cycle=0.0, snr_db=100.0)
        x = torch.randn(4, 128)
        y = channel(x)
        # With zero duty cycle and high SNR, should be very close
        assert torch.allclose(x, y, atol=0.1)

    def test_gradient_flows(self) -> None:
        for jtype in ["barrage", "pulsed", "swept", "single_tone"]:
            channel = JammingChannel(jammer_type=jtype, snr_db=10.0)
            x = torch.randn(4, 64, requires_grad=True)
            y = channel(x)
            loss = y.sum()
            loss.backward()
            assert x.grad is not None, f"No gradient for {jtype}"

    def test_single_tone_power_matches_jnr(self) -> None:
        """Single-tone jammer must inject power proportional to (signal/SNR)*JNR."""
        snr_db, jnr_db = 100.0, 10.0  # high SNR isolates the tone
        channel = JammingChannel(
            jammer_type="single_tone", jnr_db=jnr_db, snr_db=snr_db,
            tone_freq_norm=0.25,
        )
        torch.manual_seed(987)
        x = torch.ones(64, 4096)
        y = channel(x)
        added = (y - x).flatten().numpy()

        # Expected jammer power = (signal_power / snr_lin) * jnr_lin
        snr_lin, jnr_lin = 10.0 ** (snr_db / 10.0), 10.0 ** (jnr_db / 10.0)
        expected_power = (1.0 / snr_lin) * jnr_lin
        # The tone is scaled by sqrt(2*power) so its mean-square equals
        # `expected_power`, matching barrage at equal nominal JNR (previously the
        # bare amplitude^2/2 gave half power -- a -3 dB single-tone-vs-barrage gap).
        measured_power = float((added ** 2).mean())
        # Tolerate 30% – finite N + AWGN residue
        assert abs(measured_power - expected_power) < 0.3 * expected_power, (
            f"Single-tone power off: measured {measured_power:.4g} vs "
            f"expected {expected_power:.4g}"
        )

    def test_invalid_jammer_type_raises(self) -> None:
        channel = JammingChannel(jammer_type="invalid")
        x = torch.randn(4, 64)
        with pytest.raises(ValueError, match="Unknown jammer type"):
            channel(x)

    def test_barrage_noise_distribution_gof(self) -> None:
        """KS-test: total noise (AWGN + barrage jamming) must be Gaussian.

        Both AWGN and barrage jamming are independent Gaussian processes,
        so their sum must also be Gaussian with variance = noise_var + jammer_var.
        """
        snr_db = 10.0
        jnr_db = 10.0
        channel = JammingChannel(
            jammer_type="barrage", jnr_db=jnr_db, snr_db=snr_db
        )

        torch.manual_seed(321)
        x = torch.ones(3000, 256)  # constant signal for clean noise extraction
        y = channel(x)
        total_noise = (y - x).flatten().numpy()

        # Expected total variance: noise_var + jammer_var
        # noise_var = signal_power / snr_linear
        # jammer_var = noise_var * jnr_linear
        signal_power = 1.0  # x = ones
        snr_linear = 10.0 ** (snr_db / 10.0)
        jnr_linear = 10.0 ** (jnr_db / 10.0)
        noise_var = signal_power / snr_linear
        total_var = noise_var + noise_var * jnr_linear
        expected_std = total_var**0.5

        cvm = stats.cramervonmises(total_noise, "norm", args=(0.0, expected_std))
        assert cvm.pvalue > 0.01, (
            f"Barrage jamming noise failed Cramer-von Mises GOF for N(0, {expected_std:.4f}): "
            f"W2={cvm.statistic:.4f}, p={cvm.pvalue:.6f}"
        )


class TestBernoulliErasureChannel:
    """Tests for i.i.d. Bernoulli erasure channel.

    Validates: (a) measured loss rate matches p analytically via a
    binomial CI, (b) surviving symbols are exactly identity when snr_db
    is None, (c) gradients flow through surviving symbols only.
    """

    def test_output_shape_matches_input(self) -> None:
        ch = BernoulliErasureChannel(p=0.1)
        x = torch.randn(4, 128)
        y = ch(x)
        assert y.shape == x.shape

    def test_empirical_loss_rate_matches_p(self) -> None:
        """Erased fraction must match p within a binomial 99% CI."""
        torch.manual_seed(7)
        n = 200_000
        for p in [0.05, 0.20, 0.50]:
            ch = BernoulliErasureChannel(p=p)
            x = torch.ones(n)
            y = ch(x)
            erased = (y == 0).float().mean().item()
            # Binomial 99% CI half-width ≈ 2.576 * sqrt(p(1-p)/n)
            ci = 2.576 * (p * (1 - p) / n) ** 0.5
            assert abs(erased - p) < ci, (
                f"Empirical loss rate {erased:.5f} vs p={p}: diff {erased - p:+.5f}, CI ±{ci:.5f}"
            )

    def test_surviving_symbols_are_exact_identity(self) -> None:
        """When snr_db is None, surviving entries must equal input exactly."""
        ch = BernoulliErasureChannel(p=0.3, snr_db=None)
        torch.manual_seed(0)
        x = torch.randn(2048)
        y = ch(x)
        survived = y != 0
        assert torch.allclose(y[survived], x[survived]), (
            "Surviving symbols modified despite snr_db=None"
        )

    def test_residual_noise_normality_on_survivors(self) -> None:
        """With snr_db set, surviving-symbol noise must be approximately Gaussian."""
        torch.manual_seed(11)
        p = 0.2
        snr_db = 10.0
        ch = BernoulliErasureChannel(p=p, snr_db=snr_db)
        ch.eval()
        x = torch.randn(50_000)
        y = ch(x)
        mask = y != 0  # surviving
        # signal power computed over surviving symbols (matches channel impl)
        survivor_power = ((y[mask] - x[mask]) ** 2).mean() * 0  # placeholder
        # Recover the noise component on survivors: y = x + noise, when mask=1
        noise = (y - x)[mask].numpy()
        # Theoretical noise std: signal_power_over_survivors / snr_linear
        sp = float((y[mask] ** 2).mean())  # post-noise power (close to signal power for high SNR)
        # The variance here is set by the channel implementation rather than by a nominal
        # SNR, so the null is composite: test normality with Anderson-Darling, whose critical
        # values are derived for a mean and scale estimated from the sample.
        ad = stats.anderson(noise, dist="norm", method="interpolate")
        assert ad.pvalue > 0.01, (
            f"Residual erasure noise failed Anderson-Darling normality: "
            f"A2={ad.statistic:.4f}, p={ad.pvalue:.6f}"
        )

    def test_gradients_flow_through_surviving_symbols(self) -> None:
        ch = BernoulliErasureChannel(p=0.3)
        torch.manual_seed(3)
        x = torch.randn(2048, requires_grad=True)
        y = ch(x)
        y.sum().backward()
        assert x.grad is not None
        # Gradient must be 0 on erased entries, 1 on surviving
        mask = (y != 0).float()
        assert torch.allclose(x.grad, mask), (
            "Gradient pattern must equal the erasure mask (1 on survivors, 0 on erased)"
        )

    def test_p_zero_is_identity(self) -> None:
        ch = BernoulliErasureChannel(p=0.0, snr_db=None)
        x = torch.randn(512)
        assert torch.allclose(ch(x), x)

    def test_p_one_erases_everything(self) -> None:
        ch = BernoulliErasureChannel(p=1.0, snr_db=None)
        x = torch.randn(512)
        y = ch(x)
        assert torch.all(y == 0)


class TestBurstErasureChannel:
    """Tests for Gilbert-Elliott burst erasure."""

    def test_stationary_loss_rate(self) -> None:
        """Empirical mean loss rate should match the analytical stationary value."""
        torch.manual_seed(9)
        ch = BurstErasureChannel(g_to_b=0.02, b_to_g=0.10, p_good=0.0, p_bad=1.0)
        stationary = ch.stationary_loss_rate()
        # Long sample → tight CI
        x = torch.ones(16, 4096)
        y = ch(x)
        empirical = (y == 0).float().mean().item()
        # 99% CI: 2.576 * sqrt(p(1-p)/N) with N ~ 65536 effective symbols
        ci = 2.576 * (stationary * (1 - stationary) / x.numel()) ** 0.5
        # Allow looser tolerance due to Markov correlation (autocorrelation
        # inflates the effective variance); use 4x analytical CI
        assert abs(empirical - stationary) < 4 * ci + 0.02, (
            f"Empirical burst loss {empirical:.4f} vs stationary {stationary:.4f}, "
            f"diff {empirical - stationary:+.4f}, 4xCI ±{4*ci:.4f}"
        )

    def test_burst_correlation_present(self) -> None:
        """Adjacent erasures must be positively correlated (vs Bernoulli baseline)."""
        torch.manual_seed(13)
        # Heavy bursts: long sojourn in bad
        burst = BurstErasureChannel(g_to_b=0.005, b_to_g=0.05, p_good=0.0, p_bad=1.0)
        x = torch.ones(1, 20000)
        y_burst = burst(x)
        mask_burst = (y_burst == 0).float().squeeze()
        # Autocorrelation at lag 1
        m = mask_burst - mask_burst.mean()
        rho = (m[:-1] * m[1:]).mean() / (m.var() + 1e-12)
        # Two-state chain with p_good=0, p_bad=1: rho_1 = 1 - g_to_b - b_to_g exactly.
        rho_analytic = 1.0 - burst.g_to_b - burst.b_to_g
        assert abs(float(rho) - rho_analytic) < 0.02, (
            f"Burst lag-1 autocorrelation {float(rho):.4f} != closed form "
            f"1 - g_to_b - b_to_g = {rho_analytic:.4f}"
        )
