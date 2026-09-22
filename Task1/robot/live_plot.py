import json
import time

import matplotlib.pyplot as plt

DATA_FILE = "distances_live.json"
COLORS = ["red", "yellow", "green", "blue"]

fig, axes = plt.subplots(4, 1, figsize=(4.5, 5.5), sharex=True)
lines = {}

for row, name in enumerate(COLORS):
    plot_color = name if name != "yellow" else "goldenrod"
    ax = axes[row]

    (lines[(name, "true")],) = ax.plot([], [], color=plot_color, linestyle="--", label="Real")
    (lines[(name, "vision")],) = ax.plot([], [], color=plot_color, linestyle="-", label="CV")

    ax.set_ylabel(f"{name}\Distance, м")
    ax.legend(loc="upper right")

axes[-1].set_xlabel("время, с")
fig.tight_layout()

plt.show(block=False)

while plt.fignum_exists(fig.number):
    try:
        with open(DATA_FILE) as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        time.sleep(0.1)
        continue

    for name in COLORS:
        lines[(name, "true")].set_data(history[name]["t"], history[name]["true"])
        lines[(name, "vision")].set_data(history[name]["t"], history[name]["vision"])

    for ax in axes:
        ax.relim()
        ax.autoscale_view()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    time.sleep(0.1)
