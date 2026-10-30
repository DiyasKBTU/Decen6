"""Прогон истории: свойства жадного расписания."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from replay_history import simulate  # noqa: E402


def test_urgent_goes_first_and_only_after_creation():
    jobs = [(0, 3, "junk"), (1, 1, "cheat"), (5, 0, "hack")]
    out = simulate(jobs, slots=[2, 3, 6])
    assert out == {"cheat": 2, "junk": 3, "hack": 6}  # hack создан позже второго слота — ждёт свой


def test_everyone_served_and_total_wait_is_conserved():
    rng = random.Random(1)
    jobs, slots = [], []
    for i in range(500):
        created = rng.uniform(0, 1000)
        jobs.append((created, rng.randint(0, 3), f"j{i}"))
        slots.append(created + rng.expovariate(1 / 30))  # исходное расписание: каждая отвечена после создания
    out = simulate(jobs, slots)
    assert len(out) == len(jobs)
    assert all(out[j] >= c for c, _, j in jobs)
    as_is = sum(s - c for (c, _, _), s in zip(jobs, slots))
    sim = sum(out[j] - c for c, _, j in jobs)
    assert abs(as_is - sim) < 1e-6  # те же слоты: суммарное ожидание не меняется, меняется только порядок
