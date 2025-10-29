from dataclasses import dataclass

@dataclass(frozen=True)
class Summary:
    minimum_observed: float
    average_tokens_per_second: float

class ValidationError(Exception): ...

def validate(samples_tokps: list[float], minimum_rate: float) -> Summary:
    if not samples_tokps:
        raise ValidationError("Benchmark sample set cannot be empty.")
    if minimum_rate <= 0:
        raise ValidationError("Minimum throughput requirement must be positive.")
    mn = min(samples_tokps)
    if mn < minimum_rate:
        raise ValidationError(f"Observed {mn:.2f} tok/s below the minimum of {minimum_rate}.")
    avg = sum(samples_tokps) / len(samples_tokps)
    return Summary(minimum_observed=mn, average_tokens_per_second=avg)
