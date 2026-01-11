import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from matplotlib.animation import FuncAnimation
import itertools
import os


class Path:
    def __init__(self, path_id, description):
        self.id = path_id
        self.description = description


class SignalHead:
    def __init__(self, signal_id, paths):
        self.id = signal_id
        self.paths = paths

# calculating phases
def get_phases():

    p1 = Path('p1', 'North Right');
    p2 = Path('p2', 'North Straight');
    p3 = Path('p3', 'North Left')
    p4 = Path('p4', 'South Right');
    p5 = Path('p5', 'South Straight');
    p6 = Path('p6', 'South Left')
    p7 = Path('p7', 'East Right');
    p8 = Path('p8', 'East Straight');
    p9 = Path('p9', 'East Left')
    p10 = Path('p10', 'West Right');
    p11 = Path('p11', 'West Straight');
    p12 = Path('p12', 'West Left')

    all_paths = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]

    # every conflict
    path_conflicts = [
        # dik straights 90 degree
        ('p2', 'p8'), ('p2', 'p11'),
        ('p5', 'p8'), ('p5', 'p11'),

        # straight vs left
        ('p2', 'p6'),
        ('p5', 'p3'),
        ('p8', 'p12'),
        ('p11', 'p9'),

        # straight vs left 2
        ('p3', 'p8'), ('p3', 'p11'),
        ('p6', 'p8'), ('p6', 'p11'),
        ('p9', 'p2'), ('p9', 'p5'),
        ('p12', 'p2'), ('p12', 'p5'),

        # left and left
        ('p3', 'p9'), ('p3', 'p12'),
        ('p6', 'p9'), ('p6', 'p12'),

        # right left
        ('p1', 'p6'),
        ('p4', 'p3'),
        ('p7', 'p12'),
        ('p10', 'p9'),

        # going straight vs turning adjustment
        ('p2', 'p10'), ('p8', 'p1'), ('p11', 'p4'), ('p7', 'p5')
    ]
    conflict_set = set(tuple(sorted(pair)) for pair in path_conflicts)

    # all the signal heads
    signal_heads = [
        SignalHead('n1', [p1]), SignalHead('n2', [p2, p3]),
        SignalHead('n3', [p4]), SignalHead('n4', [p5, p6]),
        SignalHead('n5', [p7]), SignalHead('n6', [p8]),
        SignalHead('n7', [p9]), SignalHead('n8', [p10]),
        SignalHead('n9', [p11]), SignalHead('n10', [p12])
    ]


    # Uncomment following and comment above for Case A (Note that animation only works for Case B)
    """ 
    all_paths = [Path(f'p{i}', f'Trajectory {i}') for i in range(1, 15)]

    path_conflicts = [
        # p2
        ('p2', 'p7'), ('p2', 'p9'), ('p2', 'p10'), ('p2', 'p12'),

        # p3
        ('p3', 'p6'), ('p3', 'p10'), ('p3', 'p12'), ('p3', 'p13'),

        # p6
        ('p6', 'p9'), ('p6', 'p12'), ('p6', 'p13'),

        # p7
        ('p7', 'p9'), ('p7', 'p10'), ('p7', 'p13'),

        # p9
        ('p9', 'p13'),

        # p10
        ('p10', 'p12'),
    ]
    conflict_set = set(tuple(sorted(pair)) for pair in path_conflicts)

    signal_heads = []
    for p in all_paths:
        # Create a signal head 'nX' containing path 'pX'
        s_id = p.id.replace('p', 'n')
        signal_heads.append(SignalHead(s_id, [p]))
    """

    # finding cliques
    signal_conflict_pairs = []
    for u, v in itertools.combinations(signal_heads, 2):
        has_conflict = False

        for path_a in u.paths:
            for path_b in v.paths:
                if tuple(sorted((path_a.id, path_b.id))) in conflict_set:
                    has_conflict = True;
                    break
            if has_conflict:
                break

        if has_conflict: signal_conflict_pairs.append((u.id, v.id))

    node_ids = [s.id for s in signal_heads]
    G_compat = nx.complete_graph(node_ids)
    G_compat.remove_edges_from(signal_conflict_pairs)

    G_conflict = nx.Graph()
    G_conflict.add_nodes_from(node_ids)
    G_conflict.add_edges_from(signal_conflict_pairs)

    cliques = list(nx.find_cliques(G_compat))
    cliques.sort(key=lambda c: [int(x[1:]) for x in sorted(c)])

    print(f"Total Phases Found: {len(cliques)}")
    for i, c in enumerate(cliques):
        desc_list = []
        for nid in sorted(c, key=lambda x: int(x[1:])):
            # Find the signal head object
            sh = next(s for s in signal_heads if s.id == nid)
            # Create short text: "n2(N-Str+Left)"
            path_descs = "+".join([p.description.split(' ')[1] for p in sh.paths])
            # e.g. "Straight+Left"

            # Add direction prefix
            direction = sh.paths[0].description.split(' ')[0][0]  # 'N', 'S', 'E', 'W'
            desc_list.append(f"{nid}({direction}-{path_descs})")

        print(f"Phase {i + 1}: {', '.join(desc_list)}")

    pos = nx.circular_layout(G_compat)

    # models graph
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    nx.draw(G_conflict, pos, with_labels=True, node_color='#ffcccc', edge_color='red', font_weight='bold')
    plt.title("Conflict Graph")
    plt.subplot(1, 2, 2)
    nx.draw(G_compat, pos, with_labels=True, node_color='#ccffcc', edge_color='green', font_weight='bold')
    plt.title("Compatibility Graph")
    plt.savefig('graphs.png')
    plt.close()
    # phases graph
    num_plots = len(cliques)
    cols = 4
    rows = math.ceil(num_plots / cols)
    plt.figure(figsize=(20, 5 * rows))
    plt.suptitle("Generated Phases (Complex Intersection)", fontsize=20, y=1.02)

    for i, clique in enumerate(cliques):
        ax = plt.subplot(rows, cols, i + 1)
        # background
        nx.draw_networkx_nodes(G_compat, pos, node_color='#f0f0f0', node_size=600, ax=ax)
        nx.draw_networkx_labels(G_compat, pos, alpha=0.3, ax=ax)
        # active
        subgraph = G_compat.subgraph(clique)
        nx.draw_networkx_nodes(subgraph, pos, node_color='#00ff00', node_size=700, ax=ax)
        nx.draw_networkx_edges(subgraph, pos, edge_color='green', width=2, ax=ax)
        nx.draw_networkx_labels(subgraph, pos, font_weight='bold', ax=ax)

        # titles
        title_lines = [f"Phase {i + 1}"]
        ax.set_title("\n".join(title_lines), fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('phases.png')
    plt.close()
    print("Graphs saved.")

    return cliques, signal_heads, all_paths, signal_conflict_pairs, G_compat, node_ids

# draw road and lines
def draw_intersection(ax):
    # background
    ax.set_facecolor('#2E8B57')

    # roads
    rect_ns = patches.Rectangle((-2, -10), 4, 20, color='#555555', zorder=1)
    ax.add_patch(rect_ns)
    rect_ew = patches.Rectangle((-10, -3), 20, 6, color='#555555', zorder=1)
    ax.add_patch(rect_ew)
    rect_int = patches.Rectangle((-2, -3), 4, 6, color='#555555', zorder=1)
    ax.add_patch(rect_int)

    # lane lines
    ax.plot([0, 0], [-10, -3], color='yellow', linewidth=2, zorder=2)
    ax.plot([0, 0], [3, 10], color='yellow', linewidth=2, zorder=2)
    ax.plot([-1, -1], [3, 10], color='white', linestyle='--', linewidth=1, zorder=2)
    ax.plot([1, 1], [-3, -10], color='white', linestyle='--', linewidth=1, zorder=2)

    ax.plot([-10, -2], [0, 0], color='yellow', linewidth=2, zorder=2)
    ax.plot([2, 10], [0, 0], color='yellow', linewidth=2, zorder=2)
    ax.plot([2, 10], [1, 1], color='white', linestyle='--', linewidth=1, zorder=2)
    ax.plot([2, 10], [2, 2], color='white', linestyle='--', linewidth=1, zorder=2)
    ax.plot([-10, -2], [-1, -1], color='white', linestyle='--', linewidth=1, zorder=2)
    ax.plot([-10, -2], [-2, -2], color='white', linestyle='--', linewidth=1, zorder=2)

    # stop lines
    ax.plot([-2, 0], [3, 3], color='white', linewidth=3, zorder=2)
    ax.plot([0, 2], [-3, -3], color='white', linewidth=3, zorder=2)
    ax.plot([2, 2], [0, 3], color='white', linewidth=3, zorder=2)
    ax.plot([-2, -2], [-3, 0], color='white', linewidth=3, zorder=2)

    # north arrow
    ax.arrow(6, 6, 0, 1.2, head_width=0.6, head_length=0.6, fc='white', ec='black', width=0.2, zorder=10)
    ax.text(6, 7.8, "N", ha='center', va='bottom', fontsize=14, fontweight='bold', color='red', zorder=10)

# init traffic lights
def init_signal_lights(ax):
    lights = {}

    def add_light(x, y, label):
        circle = patches.Circle((x, y), 0.4, color='red', zorder=3, ec='black')
        ax.add_patch(circle)
        ax.text(x, y, label, color='white', fontsize=8, ha='center', va='center', fontweight='bold', zorder=4)
        return circle

    lights['n1'] = add_light(-1.5, 3, 'n1')
    lights['n2'] = add_light(-0.5, 3, 'n2')
    lights['n3'] = add_light(1.5, -3, 'n3')
    lights['n4'] = add_light(0.5, -3, 'n4')
    lights['n5'] = add_light(2, 2.5, 'n5')
    lights['n6'] = add_light(2, 1.5, 'n6')
    lights['n7'] = add_light(2, 0.5, 'n7')
    lights['n8'] = add_light(-2, -2.5, 'n8')
    lights['n9'] = add_light(-2, -1.5, 'n9')
    lights['n10'] = add_light(-2, -0.5, '10')

    return lights

# init arrows representing paths
def init_path_arrows(ax):
    arrows = {}
    labels = {}

    def add_arrow(start, end, rad, pid, color='#444444', alpha=0.3):
        # rad means curve
        arrow = patches.FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}",
                                        color=color, alpha=alpha, arrowstyle='->,head_width=0.3,head_length=0.5',
                                        mutation_scale=20, linewidth=2, zorder=2)
        ax.add_patch(arrow)

        # midpoint for label like p1
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        # adjustment for curve
        if abs(rad) > 0.1:
            if rad > 0:
                mid_x -= rad * 2; mid_y += rad * 2
            else:
                mid_x += rad * 2; mid_y -= rad * 2

        label = ax.text(mid_x, mid_y, pid, color='white', fontsize=9, fontweight='bold',
                        ha='center', va='center', bbox=dict(facecolor='black', alpha=0.0, edgecolor='none'), zorder=5)
        return arrow, label

    # NORTH
    arrows['p1'], labels['p1'] = add_arrow((-1.5, 3), (-3, 1.5), -0.3, 'p1')
    arrows['p2'], labels['p2'] = add_arrow((-0.5, 3), (-0.5, -3), 0, 'p2')
    arrows['p3'], labels['p3'] = add_arrow((-0.5, 3), (3, -0.5), 0.5, 'p3')

    # SOUTH
    arrows['p4'], labels['p4'] = add_arrow((1.5, -3), (3, -1.5), -0.3, 'p4')
    arrows['p5'], labels['p5'] = add_arrow((0.5, -3), (0.5, 3), 0, 'p5')
    arrows['p6'], labels['p6'] = add_arrow((0.5, -3), (-3, 0.5), 0.5, 'p6')

    # EAST
    arrows['p7'], labels['p7'] = add_arrow((2, 2.5), (1.5, 3), -0.3, 'p7')
    arrows['p8'], labels['p8'] = add_arrow((2, 1.5), (-2, 1.5), 0, 'p8')
    arrows['p9'], labels['p9'] = add_arrow((2, 0.5), (-0.5, -3), 0.3, 'p9')

    # WEST
    arrows['p10'], labels['p10'] = add_arrow((-2, -2.5), (-1.5, -3), -0.3, 'p10')
    arrows['p11'], labels['p11'] = add_arrow((-2, -1.5), (2, -1.5), 0, 'p11')
    arrows['p12'], labels['p12'] = add_arrow((-2, -0.5), (0.5, 3), 0.3, 'p12')

    return arrows, labels


def visualize_phases(cliques, signal_heads, all_paths):
    #cliques, signal_heads, all_paths, _, _, _ = get_phases()

    # setting up
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.axis('off')

    # drawing
    draw_intersection(ax)
    lights = init_signal_lights(ax)
    path_arrows, path_labels = init_path_arrows(ax)

    title_text = ax.text(0, 9, "", ha='center', fontsize=16, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    desc_text = ax.text(0, -9, "", ha='center', fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8))

    # animation
    def update(frame):
        phase_idx = frame % len(cliques)
        current_clique = cliques[phase_idx]

        # finding active paths
        active_paths = []
        for nid in current_clique:
            # get signal object for its paths
            sh = next((s for s in signal_heads if s.id == nid), None)
            if sh:
                for p in sh.paths:
                    active_paths.append(p.id)

        # reset
        for nid, circle in lights.items():
            circle.set_color('#cc0000')
            circle.set_edgecolor('black')

        for pid, arrow in path_arrows.items():
            arrow.set_color('#444444')
            arrow.set_alpha(0.1)  # when inactive

        for pid, label in path_labels.items():
            label.set_visible(False)  # when inactive

        # active lights and paths t green
        active_desc = []
        for nid in current_clique:
            if nid in lights:
                lights[nid].set_color('#00ff00')
                lights[nid].set_edgecolor('white')
                active_desc.append(nid)

        for pid in active_paths:
            if pid in path_arrows:
                path_arrows[pid].set_color('#00ff00')
                path_arrows[pid].set_alpha(0.9)
            if pid in path_labels:
                path_labels[pid].set_visible(True)
                path_labels[pid].set_bbox(dict(facecolor='black', alpha=0.7, edgecolor='none'))

        title_text.set_text(f"PHASE {phase_idx + 1}")
        desc_text.set_text(f"Green Lights: {', '.join(sorted(active_desc))}")

        return list(lights.values()) + list(path_arrows.values())

    output_dir = "phases_illustrated"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving {len(cliques)} phase images to '{output_dir}/'...")
    for i in range(len(cliques)):
        update(i)
        filename = os.path.join(output_dir, f"phase_{i + 1}.png")
        fig.savefig(filename, bbox_inches='tight', dpi=100)
        print(f"Saved {filename}")

    # animation
    ani = FuncAnimation(fig, update, frames=len(cliques), interval=2000, repeat=True)

    plt.show()


def run_safety_check(cliques, conflict_pairs):
    print("\nSafety verification...")
    conflict_set = set(tuple(sorted(p)) for p in conflict_pairs)

    all_safe = True
    for i, phase in enumerate(cliques):
        # checking every pair of lights in this phase
        for a, b in itertools.combinations(phase, 2):
            if tuple(sorted((a, b))) in conflict_set:
                print(f"  FAILED: Phase {i + 1} has a conflict between {a} and {b}!")
                all_safe = False

    if all_safe:
        print("PASSED: No generated phase contains conflicting signals.")
    else:
        print("Conflicts detected.")
    return all_safe


def run_maximality_check(cliques, G_compat, all_nodes):
    print("\nMaximality verification...")

    all_maximal = True
    for i, phase in enumerate(cliques):
        current_nodes = set(phase)
        outside_nodes = set(all_nodes) - current_nodes

        # tries to find a node from outside that could be added safely
        for candidate in outside_nodes:
            # checking if candidate is connected to all current nodes in compatibility graph
            is_compatible_with_all = True
            for member in phase:
                if not G_compat.has_edge(candidate, member):
                    is_compatible_with_all = False
                    break

            if is_compatible_with_all:
                print(f"FAILED: Phase {i + 1} {phase} is NOT maximal.")
                print(f"Could have added '{candidate}' but didn't.")
                all_maximal = False
                break

    if all_maximal:
        print("PASSED: All phases are maximal cliques.")
    else:
        print("FAILED: Some phases are not maximal.")
    return all_maximal


if __name__ == "__main__":
    print("Generation...")
    cliques, signal_heads, all_paths, conflict_pairs, G_compat, all_nodes = get_phases()

    print("\nValidation")
    is_safe = run_safety_check(cliques, conflict_pairs)
    is_maximal = run_maximality_check(cliques, G_compat, all_nodes)

    if is_safe and is_maximal:
        print("\nSystem is validated.")
        print("\nVisualization...")
        visualize_phases(cliques, signal_heads, all_paths)
    else:
        print("\nSystem validation failed.")