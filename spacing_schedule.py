from typing import Dict, List


def build_spacing_schedule(total_time: float,
                           start_interval: float = 2.0,
                           growth: float = 1.5,
                           num_blocks: int = 8,
                           order: List[str] = None) -> Dict[str, List[float]]:
    """
    Build an exponentially spaced sequence of trigger times alternating between
    gamma-peaks and G-switches (or any provided order), until total_time.

    Args:
      total_time: total simulation time (seconds)
      start_interval: first gap (seconds)
      growth: multiplicative growth per gap (>1)
      num_blocks: maximum number of blocks to schedule (<=0 means fill until total_time)
      order: sequence of labels to repeat (default: ['gamma', 'G'])

    Returns:
      dict with 'gamma_peaks': [times...], 'G_switches': [times...]
    """
    if order is None or len(order) == 0:
        order = ['gamma', 'G']

    gamma_peaks: List[float] = []
    G_switches: List[float] = []

    t = 0.0
    gap = max(1e-3, float(start_interval))
    block = 0
    infinite = num_blocks is None or int(num_blocks) <= 0
    while (infinite or block < int(num_blocks)) and t < total_time:
        t += gap
        label = order[block % len(order)]
        if t >= total_time:
            break
        if label.lower().startswith('g') and label.lower() != 'gamma':
            G_switches.append(t)
        else:
            gamma_peaks.append(t)
        gap *= max(1.0, float(growth))
        block += 1

    return {
        'gamma_peaks': gamma_peaks,
        'G_switches': G_switches
    } 