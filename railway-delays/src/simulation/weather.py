"""Daily weather for the simulator (SIMULATED unless a real source is plugged in).

`WeatherSource` is the interface the simulator uses. To use real data (for example Open-Meteo),
write a class with the same `state(day)` method and `source` attribute and pass it to the run.
"""
import datetime
import random

EPOCH = datetime.date(2025, 1, 1)
STATES = ("clear", "fog", "rain")


def day_to_date(day):
    return EPOCH + datetime.timedelta(days=int(day))


class WeatherSource:
    """Interface: state(day) -> "clear" | "fog" | "rain", where day 0 is 2025-01-01."""

    source = "abstract"

    def state(self, day):
        raise NotImplementedError


class FixedWeather(WeatherSource):
    """Same state every day (used in tests and sensitivity runs)."""

    source = "fixed"

    def __init__(self, state="clear"):
        self._state = state

    def state(self, day):
        return self._state


class SyntheticWeather(WeatherSource):
    """Markov chain over days with monthly fog/rain probabilities, started far enough before the
    first simulated day that a given date has the same state whatever range is simulated."""

    source = "synthetic"
    ANCHOR = -120

    def __init__(self, cfg, seed, anchor=None):
        self.w = cfg["weather"]
        self._rnd = random.Random(f"{seed}-weather")
        self._states = {}
        self._last = None
        self.anchor = self.ANCHOR if anchor is None else int(anchor)
        self._next = self.anchor

    def _probs(self, day):
        m = day_to_date(day).month
        pf = self.w["fog_probability"].get(m, 0.0)
        pr = self.w["rain_probability"].get(m, self.w["rain_default"])
        return pf, pr

    def _extend(self, upto):
        pers = self.w["persistence"]
        while self._next <= upto:
            pf, pr = self._probs(self._next)
            # persistence: the previous day repeats more often, but only in months where it can occur
            p_fog = pf * (1 - pers) + (pers if (self._last == "fog" and pf > 0) else 0.0)
            p_rain = pr * (1 - pers) + (pers if (self._last == "rain" and pr > 0) else 0.0)
            u = self._rnd.random()
            if u < p_fog:
                s = "fog"
            elif u < p_fog + p_rain:
                s = "rain"
            else:
                s = "clear"
            self._states[self._next] = s
            self._last = s
            self._next += 1

    def state(self, day):
        day = int(day)
        if day < self.anchor:
            raise ValueError("day before the weather anchor")
        self._extend(day)
        return self._states[day]
