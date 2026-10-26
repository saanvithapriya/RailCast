"""Ground-truth cause codes written to the simulated observations."""
NONE = 0
INCIDENT = 1        # injected section incident
DISRUPTION = 2      # block or signal failure
WEATHER = 3         # weather slowdown
SECTION_WAIT = 4    # headway / following another train
YIELD = 5           # yielding to a higher-priority train
FREIGHT = 6         # freight blocker
RAKE = 7            # rake reuse (late arrival of the feeding train)
HALT_EXT = 8        # halt extension
ENTRY = 9           # unknown incoming delay of a train entering mid-route
INHERITED = 10      # delay carried from upstream, nothing new added
RECOVERY = -1       # negative minutes: slack used / halt shortened

NAMES = {
    0: "none", 1: "incident", 2: "disruption", 3: "weather", 4: "section_wait", 5: "yield",
    6: "freight", 7: "rake_reuse", 8: "halt_extension", 9: "entry_delay", 10: "inherited", -1: "recovery",
}
