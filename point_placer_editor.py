def temp():
    import matplotlib.pyplot as plt
    import math

    # store points as list of (x, y)
    points = []

    fig, ax = plt.subplots()
    ax.set_title("Left-click: add point | Right-click: remove nearest | 'p': print all | 'c': clear")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    scatter = ax.scatter([], [])  # initial empty scatter

    def redraw():
        """Redraw scatter from points list."""
        if points:
            xs, ys = zip(*points)
        else:
            xs, ys = [], []
        scatter.set_offsets(list(zip(xs, ys)))
        fig.canvas.draw_idle()

    def add_point(x, y):
        points.append((x, y))
        print(f"[{x:.6f}, {y:.6f}],")
        redraw()

    def remove_nearest(x, y, max_dist=None):
        """Remove the nearest point to (x, y). If max_dist set, only remove if within that distance."""
        if not points:
            print("No points to remove.")
            return
        best_idx = None
        best_d2 = None
        for i, (px, py) in enumerate(points):
            d2 = (px - x)**2 + (py - y)**2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_idx = i
        if max_dist is not None and best_d2 is not None and best_d2 > max_dist**2:
            print("No point near the click to remove.")
            return
        removed = points.pop(best_idx)
        print(f"Removed: ({removed[0]:.6f}, {removed[1]:.6f})")
        redraw()

    def on_click(event):
        # only respond to clicks inside the axes
        if event.inaxes != ax:
            return
        x, y = event.xdata, event.ydata
        # button: 1=left, 2=middle, 3=right (backend dependent on some platforms)
        if event.button == 1:
            add_point(x, y)
        elif event.button == 3:
            # remove nearest point; use a small threshold so accidental far-right-clicks don't delete
            remove_nearest(x, y, max_dist=0.05 * max(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0]))

    def on_key(event):
        if event.key == 'p':
            if not points:
                print("No points.")
            else:
                print("All points:")
                for i, (x, y) in enumerate(points):
                    print(f"  {i}: ({x:.6f}, {y:.6f})")
        elif event.key == 'c':
            points.clear()
            redraw()
            print("Cleared all points.")
        elif event.key == 'q':
            print("Quit requested (closing window).")
            plt.close(fig)

    # connect events
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # optionally show grid
    ax.grid(True)

    print("Canvas ready. Left-click to add points. Right-click to remove nearest. Press 'p' to print all, 'c' to clear, 'q' to quit.")
    plt.show()

temp()