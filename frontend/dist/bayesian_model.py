import numpy as np
import time
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

# --- Math Helpers to replace Scipy ---

def norm_pdf(x, mu, sigma):
    """PDF of normal distribution"""
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def brentq(f, a, b, xtol=1e-6, max_iter=100):
    """
    Find a root of the function f in the interval [a, b] using the bisection method.
    (Simplified replacement for scipy.optimize.brentq)
    """
    if f(a) * f(b) >= 0:
        # If signs are same, we can't guarantee a root with bisection.
        # Fallback: try to find a sign change or just return the one closer to 0
        if abs(f(a)) < abs(f(b)):
            return a
        return b
        
    for _ in range(max_iter):
        c = (a + b) / 2
        if (b - a) / 2 < xtol:
            return c
        if f(c) == 0:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

# -------------------------------------

@dataclass
class Experience:
    """Represents a single experience with its distribution"""
    id: int
    objective_mean: float  # The mean of the experience distribution
    objective_sigma: float  # The original sigma (emotional charge: low σ = high charge)
    stored_mean: float  # The interpreted/squeezed mean
    stored_sigma: float  # The stored sigma (may differ if squeezed)
    timestamp: float  # Unix timestamp when experience was added
    squeezing_cost: float = 0.0  # Cost paid to squeeze this experience
    was_squeezed: bool = False
    meditation_level: float = 0.0  # How much this experience has been meditated on (0.0 to 1.0)
    original_stored_mean: float | None = None  # The stored_mean before any meditation (for interpolation)
    creation_brahma_viharas_level: float = 0.0  # The brahma viharas level when this experience was created
    target_meditation_level: float = 0.0  # Target meditation level for manual meditation (animated)
    manual_meditation_start_time: float | None = None  # When manual meditation was triggered
    is_manual_meditation: bool = False  # Whether this meditation is from manual button (increases cultivation)
    
    @property
    def tau(self) -> float:
        """Precision: 1/σ²"""
        return 1.0 / (self.stored_sigma ** 2)

    def to_dict(self) -> Dict:
        import time
        time_elapsed = time.time() - self.timestamp

        return {
            'id': self.id,
            'objective_mean': float(self.objective_mean),
            'objective_sigma': float(self.objective_sigma),
            'stored_mean': float(self.stored_mean),
            'stored_sigma': float(self.stored_sigma),
            'tau': float(self.tau),
            'timestamp': self.timestamp,
            'time_elapsed': float(time_elapsed),  # Seconds since experience was added
            'squeezing_cost': float(self.squeezing_cost),
            'squeeze_distance': float(abs(self.objective_mean - self.stored_mean)),  # How far it was squeezed
            'was_squeezed': self.was_squeezed,
            'meditation_level': float(self.meditation_level)
        }


class BayesianBeliefModel:
    """
    Models belief formation through precision-weighted Bayesian inference
    where precision (tau) represents emotional/karmic charge
    """
    
    def __init__(self):
        self.experiences: List[Experience] = []
        self.experience_counter: int = 0
        self.brahma_viharas_level: float = 0.0  # Cultivation level (0-1), single counter
        self.meditation_practice_count: int = 0  # Track meditation sessions for skill growth

        # Prior parameters (extremely weak prior - wide initial belief)
        self.prior_mu = 0.0
        self.prior_tau = 0.01  # Very weak prior (σ = 10.0, 95% CI = ±19.6)
        
    def get_posterior_params(self) -> Tuple[float, float]:
        """
        Calculate posterior mean and sigma using precision-weighted update
        P(belief | experiences) ∝ P(experiences | belief) × P(belief)

        Meditation on experiences reduces their precision (tau), making them
        have less "pull" on the posterior. Fully meditated experiences barely
        influence the posterior at all.
        """
        if not self.experiences:
            prior_sigma = 1.0 / np.sqrt(self.prior_tau)
            return self.prior_mu, prior_sigma

        # Standard Bayesian update with meditation-adjusted tau
        sum_weighted_values = self.prior_tau * self.prior_mu
        current_tau_sum = self.prior_tau

        for exp in self.experiences:
            # Use meditation-adjusted tau (meditation reduces precision)
            effective_tau = self.get_effective_tau(exp)

            sum_weighted_values += effective_tau * exp.stored_mean
            current_tau_sum += effective_tau

        posterior_mu = sum_weighted_values / current_tau_sum
        posterior_sigma = 1.0 / np.sqrt(current_tau_sum)

        return posterior_mu, posterior_sigma

    def calculate_all_experience_reduction(self, meditated_exp: Experience) -> float:
        """
        Calculate the tau reduction factor for ALL experiences such that at 100% meditation,
        the posterior CI tightly covers all experiences while staying informed by the data.

        Uses root-finding to find the tau that makes the CI just cover all experiences.

        Returns: reduction_factor (0 to 1), where 1 = no reduction, smaller = more reduction
        """
        # Get all relevant experiences
        relevant_exps = [exp for exp in self.experiences if exp.id <= meditated_exp.id]
        exp_means = [exp.objective_mean for exp in relevant_exps]
        min_mean = min(exp_means)
        max_mean = max(exp_means)
        n_exps = len(relevant_exps)

        # Function to check if all experiences fit with a given reduction factor
        def check_coverage(reduction_factor: float) -> float:
            """
            Returns negative if experiences don't fit, positive if they do with margin.
            Zero at the boundary.
            """
            if reduction_factor < 1e-10:
                reduction_factor = 1e-10

            # Calculate posterior with this reduction
            sum_weighted = self.prior_tau * self.prior_mu
            sum_tau = self.prior_tau

            for exp in relevant_exps:
                tau = exp.tau * reduction_factor
                sum_weighted += tau * exp.objective_mean
                sum_tau += tau

            mu = sum_weighted / sum_tau
            sigma = 1.0 / np.sqrt(sum_tau)
            ci_lower = mu - 1.96 * sigma
            ci_upper = mu + 1.96 * sigma

            # Check if all experiences fit with margin
            margin = 0.05
            min_distance = float('inf')

            for exp_mean in exp_means:
                dist_to_lower = exp_mean - (ci_lower + margin)
                dist_to_upper = (ci_upper - margin) - exp_mean

                # Minimum of how far inside the CI this experience is
                # Negative means outside
                min_distance = min(min_distance, dist_to_lower, dist_to_upper)

            return min_distance

        # Check if already fits with no reduction
        if check_coverage(1.0) >= 0:
            return 1.0

        # Check if impossible even with tiny tau
        if check_coverage(1e-10) < 0:
            return 1e-10

        # Use root-finding to find the boundary
        try:
            fitting_reduction: float = brentq(check_coverage, 1e-10, 1.0, xtol=1e-6)  # type: ignore
            return fitting_reduction
        except ValueError:
            # If fails, use small reduction
            return 1e-6

    def get_effective_tau(self, experience: Experience) -> float:
        """
        Get tau (precision) adjusted by meditation on this or LATER experiences.

        When any experience is meditated on, ALL experiences at or before it have
        their tau reduced to allow the posterior to be wide enough to include everything.

        At 100% meditation, all relevant experiences have reduced tau such that the
        posterior CI encompasses all their objective_means.
        """
        base_tau = experience.tau

        # Check if THIS experience or any LATER experiences are being meditated on
        # Apply the maximum reduction from all applicable meditations
        max_reduction_factor = 1.0

        for later_exp in self.experiences:
            # If a later exp is meditated on, it affects this exp
            # If THIS exp is meditated on, it affects itself
            if later_exp.id >= experience.id and later_exp.meditation_level > 0:
                # Calculate reduction factor at 100% meditation
                fitting_reduction = self.calculate_all_experience_reduction(later_exp)

                # IMPORTANT: Linearly interpolate SIGMA (not tau) for linear CI width scaling
                # sigma = 1/sqrt(tau), so we need to:
                # 1. Convert tau to sigma at both endpoints
                # 2. Interpolate sigma
                # 3. Convert back to tau

                sigma_original = 1.0 / np.sqrt(base_tau)  # sigma at meditation=0
                tau_at_100 = base_tau * fitting_reduction
                sigma_at_100 = 1.0 / np.sqrt(tau_at_100)  # sigma at meditation=1

                # Linearly interpolate sigma based on meditation level
                sigma_interpolated = sigma_original + (sigma_at_100 - sigma_original) * later_exp.meditation_level

                # Convert back to tau reduction factor
                tau_interpolated = 1.0 / (sigma_interpolated ** 2)
                reduction = tau_interpolated / base_tau

                # Take the most restrictive (smallest) reduction
                max_reduction_factor = min(max_reduction_factor, reduction)

        return base_tau * max_reduction_factor

    def calculate_squeezing_cost(self, objective_mean: float, objective_sigma: float) -> float:
        """
        Calculate cost of squeezing an experience to fit within current posterior

        Cost is based on how much the posterior distribution gets "reactivated" 
        (perturbed/tightened) when the squeezed experience is added.

        The cost reflects:
        1. The distance the experience needs to be shifted (mean_shift)
        2. How much the posterior's precision (tau) increases after adding it
        
        A high-precision experience that gets squeezed far creates a tight, 
        reactive posterior = high suffering.

        Returns: squeezing cost (0 if within 95% CI)
        """
        # Get current posterior before adding this experience
        posterior_mu_before, posterior_sigma_before = self.get_posterior_params()
        posterior_tau_before = 1.0 / (posterior_sigma_before ** 2)

        # 95% confidence interval of current posterior
        ci_lower = posterior_mu_before - 1.96 * posterior_sigma_before
        ci_upper = posterior_mu_before + 1.96 * posterior_sigma_before

        # If experience mean is within CI, no squeezing needed
        if ci_lower <= objective_mean <= ci_upper:
            return 0.0

        # Outside CI: need to squeeze it to the nearest boundary
        if objective_mean < ci_lower:
            stored_mean = ci_lower
            mean_shift = ci_lower - objective_mean
        else:
            stored_mean = ci_upper
            mean_shift = objective_mean - ci_upper

        # Calculate what the posterior would be AFTER adding the squeezed experience
        # This shows how much the posterior gets "reactivated" (tighter/more certain)
        exp_tau = 1.0 / (objective_sigma ** 2)
        
        # Bayesian update: tau_after = tau_before + exp_tau
        posterior_tau_after = posterior_tau_before + exp_tau
        
        # Cost = squared distance shifted × how much the posterior tightens
        # The posterior's increased precision (tau_after) measures the "reactivation"
        cost = (mean_shift ** 2) * posterior_tau_after

        return cost

    def should_squeeze(self, objective_mean: float, objective_sigma: float) -> Tuple[bool, float, float]:
        """
        Decide whether experience needs to be squeezed

        Returns: (should_squeeze, stored_mean, stored_sigma)
        """
        posterior_mu, posterior_sigma = self.get_posterior_params()

        # 95% confidence interval
        ci_lower = posterior_mu - 1.96 * posterior_sigma
        ci_upper = posterior_mu + 1.96 * posterior_sigma

        # If within CI: accept as-is
        if ci_lower <= objective_mean <= ci_upper:
            return False, objective_mean, objective_sigma

        # Outside CI: squeeze to nearest boundary
        if objective_mean < ci_lower:
            stored_mean = ci_lower
        else:
            stored_mean = ci_upper

        return True, stored_mean, objective_sigma
    
    def add_experience(self, value: float, sigma: float = 1.0):
        """
        Add a new experience with specified mean and sigma (emotional charge)

        Args:
            value: The objective mean of the experience
            sigma: The sigma/uncertainty of the experience (low = high emotional charge)
        """
        # Determine if we need to squeeze and calculate cost
        should_squeeze_flag, stored_mean, stored_sigma = self.should_squeeze(value, sigma)
        cost = self.calculate_squeezing_cost(value, sigma) if should_squeeze_flag else 0.0

        exp = Experience(
            id=self.experience_counter,
            objective_mean=value,
            objective_sigma=sigma,
            stored_mean=stored_mean,
            stored_sigma=stored_sigma,
            timestamp=time.time(),  # Store actual unix timestamp
            squeezing_cost=cost,
            was_squeezed=should_squeeze_flag,
            meditation_level=0.0,  # Start with no meditation
            original_stored_mean=stored_mean,  # Track original for interpolation
            creation_brahma_viharas_level=self.brahma_viharas_level  # Capture current cultivation level
        )

        self.experiences.append(exp)
        self.experience_counter += 1

        # All experiences (both squeezed and non-squeezed) will be meditated 
        # gradually over time (see apply_time_based_meditation)

        return exp.to_dict()
    
    def meditate_on_experience(self, experience_id: int, meditation_level: float, increase_cultivation: bool = True):
        """
        Meditate on an experience to release rigidity and expand the model.

        When you meditate on an experience (especially a squeezed one), you realize
        that ALL experiences (old and new) are valid data points about reality.
        None should rigidly constrain the model.

        Meditation reduces the tau (precision/charge) of ALL experiences at or before
        this one, allowing the posterior to expand and accommodate the full range of
        experiences that have occurred.

        At 100% meditation, ALL affected experiences have their tau reduced just enough
        that the posterior's 95% CI includes all their objective_means. The experiences
        still contribute information about what's possible, but no longer create a
        rigid, narrow model that rejects new data.

        Meditation is one-way (can only increase, never decrease).

        Args:
            experience_id: The ID of the experience to meditate on
            meditation_level: New meditation level (0.0 to 1.0), must be >= current level
            increase_cultivation: Whether to increase cultivation level (True for manual, False for automatic)
        """
        # Find the experience
        exp = next((e for e in self.experiences if e.id == experience_id), None)
        if exp is None:
            raise ValueError(f"Experience with id {experience_id} not found")

        # Clamp meditation level
        meditation_level = max(0.0, min(1.0, meditation_level))

        # Can only increase meditation (one-way)
        if meditation_level <= exp.meditation_level:
            return  # No change

        # Calculate squeezing cost BEFORE meditation (only if increasing cultivation)
        cost_before = 0.0
        if increase_cultivation:
            cost_before = sum(
                self.calculate_squeezing_cost(e.objective_mean, e.objective_sigma)
                for e in self.experiences
            )

        # Update meditation level
        exp.meditation_level = meditation_level

        # Meditating on this experience releases ALL older experiences
        # Move their stored_mean back toward objective_mean
        for old_exp in self.experiences:
            if old_exp.id < exp.id and old_exp.was_squeezed and old_exp.original_stored_mean is not None:
                # Interpolate from original squeezed mean to objective mean
                old_exp.stored_mean = old_exp.original_stored_mean * (1 - meditation_level) + old_exp.objective_mean * meditation_level

        # Also release the meditated experience itself if it was squeezed
        if exp.was_squeezed and exp.original_stored_mean is not None:
            # Interpolate from original squeezed mean to objective mean
            exp.stored_mean = exp.original_stored_mean * (1 - meditation_level) + exp.objective_mean * meditation_level

        # Check if meditation allows unsqueezing (updates was_squeezed flag)
        self._check_unsqueeze(exp)

        # Calculate squeezing cost AFTER meditation and increase cultivation (only for manual meditation)
        if increase_cultivation:
            cost_after = sum(
                self.calculate_squeezing_cost(e.objective_mean, e.objective_sigma)
                for e in self.experiences
            )

            # Track the reduction in squeezing cost and increase cultivation
            # Each 100 points of squeezing reduced increases cultivation by 0.01 (1%)
            cost_reduction = max(0.0, cost_before - cost_after)
            cultivation_increase = np.floor(cost_reduction / 100.0) * 0.01
            self.brahma_viharas_level = min(1.0, self.brahma_viharas_level + cultivation_increase)

        # Track meditation practice count (for statistics/display)
        self.meditation_practice_count += 1

    def meditate_on_latest(self, meditation_level: float):
        """
        Convenience method to set a target meditation level for the latest experience.

        Instead of immediately applying meditation, this sets a target that will be
        reached gradually through apply_time_based_meditation(). This creates a smooth
        animation similar to automatic cultivation-based meditation.
        """
        if not self.experiences:
            return

        latest_exp = self.experiences[-1]
        
        # Clamp meditation level
        meditation_level = max(0.0, min(1.0, meditation_level))
        
        # Can only increase target meditation (one-way)
        if meditation_level <= latest_exp.target_meditation_level:
            return
        
        # Set the target and mark when manual meditation started
        latest_exp.target_meditation_level = meditation_level
        if latest_exp.manual_meditation_start_time is None:
            latest_exp.manual_meditation_start_time = time.time()
        
        # Mark this as manual meditation (should increase cultivation)
        latest_exp.is_manual_meditation = True

    def set_brahma_viharas_level(self, level: float):
        """
        Manually set the brahma viharas cultivation level (0.0 to 1.0).

        This allows manual override of the cultivation level. Meditation practice
        can only INCREASE this level from the manually set value, never decrease it.
        """
        self.brahma_viharas_level = max(0.0, min(1.0, level))

    def apply_time_based_meditation(self, delay_seconds: float = 0.5, decay_duration: float = 5.0, 
                                   manual_duration: float = 2.0):
        """
        Apply gradual meditation to ALL experiences based on elapsed time.

        This handles two types of meditation:
        1. Automatic cultivation-based meditation (from brahma viharas level)
        2. Manual meditation (from clicking the meditate button)

        Both animate smoothly over time for a consistent user experience.

        Args:
            delay_seconds: When to START auto-meditation (5 seconds)
            decay_duration: How long the auto-meditation takes (5 seconds)
            manual_duration: How long manual meditation animation takes (2 seconds)
        """
        current_time = time.time()

        for exp in self.experiences:
            target_meditation = 0.0
            
            # 1. Calculate automatic cultivation-based meditation target
            if exp.creation_brahma_viharas_level > 0:
                time_elapsed = current_time - exp.timestamp

                # Start meditating after delay, increase gradually over decay_duration
                if time_elapsed >= delay_seconds:
                    # Calculate progress through decay (0 to 1)
                    decay_progress = min(1.0, (time_elapsed - delay_seconds) / decay_duration)

                    # Target meditation increases gradually to the cultivation level WHEN THIS EXPERIENCE WAS CREATED
                    auto_target = exp.creation_brahma_viharas_level * decay_progress
                    target_meditation = max(target_meditation, auto_target)
            
            # 2. Calculate manual meditation target (if user clicked meditate button)
            if exp.manual_meditation_start_time is not None:
                time_since_manual = current_time - exp.manual_meditation_start_time
                
                # Animate manual meditation over manual_duration
                manual_progress = min(1.0, time_since_manual / manual_duration)
                
                # Interpolate from current level to target level
                manual_meditation = exp.meditation_level + (exp.target_meditation_level - exp.meditation_level) * manual_progress
                target_meditation = max(target_meditation, manual_meditation)
            
            # Apply the highest target (only increase meditation, never decrease)
            if target_meditation > exp.meditation_level:
                # Check if this is manual meditation (should increase cultivation)
                should_increase_cultivation = exp.is_manual_meditation
                self.meditate_on_experience(exp.id, target_meditation, increase_cultivation=should_increase_cultivation)
                
                # Once manual meditation is complete, reset the flag
                if exp.is_manual_meditation and target_meditation >= exp.target_meditation_level:
                    exp.is_manual_meditation = False

    def _check_unsqueeze(self, exp: Experience):
        """
        Check if an experience can unsqueeze due to meditation.

        If the objective_mean now fits within the current posterior CI
        (because meditation has widened it), unsqueeze the experience.
        """
        # Skip if already unsqueezed
        if not exp.was_squeezed:
            return

        # Get current posterior
        posterior_mu, posterior_sigma = self.get_posterior_params()
        ci_lower = posterior_mu - 1.96 * posterior_sigma
        ci_upper = posterior_mu + 1.96 * posterior_sigma

        # Can the objective_mean fit now?
        if ci_lower <= exp.objective_mean <= ci_upper:
            # Yes! Unsqueeze it
            exp.stored_mean = exp.objective_mean
            exp.was_squeezed = False
            exp.squeezing_cost = 0.0

    def get_posterior_distribution(self, x_range: np.ndarray | None = None) -> Dict:
        """Get the posterior distribution for visualization"""
        if x_range is None:
            x_range = np.linspace(-10, 10, 200)

        # Get current posterior
        mu, sigma = self.get_posterior_params()
        tau = 1.0 / (sigma ** 2)
        pdf = norm_pdf(x_range, mu, sigma)

        # 95% confidence interval
        ci_lower = mu - 1.96 * sigma
        ci_upper = mu + 1.96 * sigma

        return {
            'x': x_range.tolist(),
            'pdf': pdf.tolist(),
            'mu': float(mu),
            'sigma': float(sigma),
            'tau': float(tau),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper)
        }
    
    def get_experience_distributions(self, x_range: np.ndarray | None = None) -> List[Dict]:
        """Get probability distributions for all experiences"""
        if x_range is None:
            x_range = np.linspace(-15, 15, 300)

        distributions = []
        posterior_mu, posterior_sigma = self.get_posterior_params()
        ci_lower = posterior_mu - 1.96 * posterior_sigma
        ci_upper = posterior_mu + 1.96 * posterior_sigma

        for exp in self.experiences:
            # Get meditation-adjusted tau, then convert to sigma for plotting
            effective_tau = self.get_effective_tau(exp)
            effective_sigma = 1.0 / np.sqrt(effective_tau)
            base_sigma = exp.stored_sigma  # Without meditation adjustment
            
            # Check if currently squeezed
            currently_squeezed = not (ci_lower <= exp.objective_mean <= ci_upper)
            current_squeezing_cost = self.calculate_squeezing_cost(exp.objective_mean, exp.objective_sigma)

            # Stored distribution with meditation adjustment
            pdf_stored = norm_pdf(x_range, exp.stored_mean, effective_sigma)

            # Objective distribution
            pdf_objective = norm_pdf(x_range, exp.objective_mean, effective_sigma)

            distributions.append({
                'id': exp.id,
                'x': x_range.tolist(),
                'pdf_stored': pdf_stored.tolist(),
                'pdf_objective': pdf_objective.tolist(),
                'stored_mean': float(exp.stored_mean),
                'objective_mean': float(exp.objective_mean),
                'was_squeezed': exp.was_squeezed,
                'currently_squeezed': currently_squeezed,
                'squeezing_cost': float(current_squeezing_cost),
                'original_squeezing_cost': float(exp.squeezing_cost),
                'effective_sigma': float(effective_sigma),
                'base_sigma': float(base_sigma),
                'meditation_level': float(exp.meditation_level)
            })

        return distributions
    
    def get_suffering_metrics(self) -> Dict:
        """Calculate all suffering-related metrics"""
        # Calculate current squeezing costs
        individual_costs = []
        for exp in self.experiences:
            current_cost = self.calculate_squeezing_cost(exp.objective_mean, exp.objective_sigma)
            individual_costs.append({
                'id': exp.id,
                'cost': float(current_cost),
                'original_cost': float(exp.squeezing_cost)
            })

        # Current total squeezing cost
        current_total_cost = sum(c['cost'] for c in individual_costs)

        return {
            'individual_costs': individual_costs,
            'total_squeezing_cost': float(current_total_cost),
            'historical_squeezing_cost': float(sum(exp.squeezing_cost for exp in self.experiences))
        }
    
    def get_state(self) -> Dict:
        """Get complete state for dashboard"""
        # Apply time-based auto-meditation before returning state
        # Starts at 5s, completes by 10s (5s delay + 5s decay)
        self.apply_time_based_meditation(delay_seconds=0.5, decay_duration=5.0)

        x_range = np.linspace(-10, 10, 200)

        return {
            'experiences': [exp.to_dict() for exp in self.experiences],
            'posterior': self.get_posterior_distribution(x_range),
            'experience_distributions': self.get_experience_distributions(x_range),
            'suffering_metrics': self.get_suffering_metrics(),
            'brahma_viharas_level': float(self.brahma_viharas_level),
            'meditation_practice_count': self.meditation_practice_count
        }
