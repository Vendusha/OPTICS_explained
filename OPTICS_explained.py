import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.lines import Line2D  # For legend line representations
from scipy.spatial.distance import euclidean
import scipy.cluster.hierarchy as sch  # For hierarchical clustering
import heapq
import os
import imageio
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # For creating inset plots
import math  # For trigonometric functions

# Define the points with labels
labels = [chr(i) for i in range(65, 81)]  # Labels A to P
points = np.array([
   [5, 35],    # A -0
   [5, 40],    # B -1
   [18, 33],   # C -2
   [13, 39],   # D -3
   [35, 40],   # E -4
   [30, 40],   # F -5
   [45, 38],   # G -6
   [43, 42],   # H -7
   [50, 10],   # I -8
   [70, 10],   # J -9
   [70, 20],   # K -10
   [80, 20],   # L -11
   [60, 40],   # M -12
   [65, 30],   # N -13
   [60, 45],   # O -14
   [65, 35],   # P -15
])

# Parameters for OPTICS
min_samples = 2
max_eps = 20  # Adjusted based on your dataset

# Function to compute core distance
def core_distance(point, neighbors, min_samples):
    if len(neighbors) < min_samples:
        return np.inf
    distances = sorted([euclidean(point, neighbor) for neighbor in neighbors])
    return distances[min_samples - 1]

# Function to get neighbors within max_eps
def get_neighbors(point_idx, points, max_eps):
    neighbors = []
    for idx, p in enumerate(points):
        if idx != point_idx and euclidean(points[point_idx], p) <= max_eps:
            neighbors.append(p)
    return neighbors

# Initialize variables for OPTICS
N = len(points)
processed = [False] * N
reachability = [None] * N
ordering = []
core_distances = [0.0] * N

# Initialize reachability plot data
reach_plot_y = []

# Create output directory
output_dir = "optics_steps"
os.makedirs(output_dir, exist_ok=True)

# Function to create the legend
def create_legend(ax):
    """
    Creates a legend for the OPTICS visualization, including point categories and distance circles.

    Parameters:
    - ax: The matplotlib axis where the legend will be added.
    """
    # Define legend elements for point categories
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Processed Points'),
        Patch(facecolor='red', edgecolor='black', label='Current Point'),
        Patch(facecolor='orange', edgecolor='black', label='Core point ($\in$ Seed)'),
        Patch(facecolor='blue', edgecolor='black', label='Seed point'),
        Patch(facecolor='yellow', edgecolor='black', label='Unprocessed Points')
    ]
    
    # Define legend elements for distance circles using Line2D
    distance_elements = [
        Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='$\epsilon_{core}$'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='$\epsilon_{max}$')
    ]
    
    # Combine both lists
    legend_elements += distance_elements
    
    # Add the legend to the axis
    ax.legend(handles=legend_elements, loc='upper right', prop={'size': 15})

# Implement a priority queue with heapq and track its elements using lazy deletion
class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def add_point(self, point_idx, distance):
        heapq.heappush(self.elements, (distance, point_idx))
    
    def pop_point(self, processed):
        while self.elements:
            distance, point_idx = heapq.heappop(self.elements)
            if not processed[point_idx]:
                return point_idx
        return None
    
    def is_empty(self):
        return len(self.elements) == 0
    
    def get_seed_list(self, processed):
        # Returns a list of tuples (label, reachability distance) without modifying the queue
        seed = sorted([(labels[idx], reachability[idx]) for (dist, idx) in self.elements if not processed[idx]], key=lambda x: x[1])
        return seed

# Function to assign clusters based on reachability distances using hierarchical clustering
def assign_clusters(points, reachability, min_samples, max_eps):
    """
    Assign clusters based on reachability distances using hierarchical clustering.

    Parameters:
    - points: Numpy array of point coordinates.
    - reachability: List of reachability distances.
    - min_samples: Minimum number of samples to form a core point.
    - max_eps: Maximum epsilon distance.

    Returns:
    - clusters: List of cluster labels for each point.
    """
    max_eps = 10
    # Perform hierarchical clustering on the original points using 'single' linkage
    Z = sch.linkage(points, method='single')
    
    # Determine cluster labels using a distance threshold.
    clusters = sch.fcluster(Z, t=max_eps, criterion='distance')
    
    return clusters

# Function to map each unique cluster to a distinct color
def map_clusters_to_colors(clusters):
    """
    Map each unique cluster to a distinct color.

    Parameters:
    - clusters: List of cluster labels for each point.

    Returns:
    - cluster_colors: Dictionary mapping cluster labels to colors.
    """
    unique_clusters = np.unique(clusters)
    # Choose a colormap with enough distinct colors
    cmap = plt.get_cmap('tab20')  # Adjust if you have more than 20 clusters
    colors = cmap.colors[:len(unique_clusters)]
    cluster_colors = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
    return cluster_colors

# Initialize the plot
def initialize_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("OPTICS Algorithm Step 0")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Plot all points as larger yellow with black circles
    for idx, (x, y) in enumerate(points):
        ax.scatter(x, y, c='yellow', edgecolors='black', s=600)  # Increased size
        ax.text(x, y, labels[idx], fontsize=12, ha='center', va='center')
    
    # Add the legend
    create_legend(ax)
    
    # Add the inset for the reachability plot
    axins = inset_axes(ax, width="30%", height="30%", loc='lower left', borderpad=5)
    axins.set_title("Reachability Plot", fontsize=12)
    axins.set_xlabel("Order", fontsize=12)
    axins.set_ylabel("Reach Dist", fontsize=12)
    axins.tick_params(axis='both', which='major', labelsize=12)
    
    # Initially, no bars to plot
    axins.bar([], [])
    
    # Save the initial plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"step_0.png"))
    plt.close()

# Update the plot at each step
def update_plot(step, current_idx, new_ordering, reach_plot_y, seed_list):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title(f"OPTICS Algorithm Step {step}")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 50)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_xlabel("X", fontsize=15)
    ax1.set_ylabel("Y", fontsize=15)

    # Set the tick label size
    ax1.tick_params(axis='both', labelsize=15)

    
    # Draw all points with updated styles
    for idx, (x, y) in enumerate(points):
        if idx == current_idx:
            color = 'red'    # Current point has highest priority
        elif processed[idx]:
            color = 'green'  # Processed points
        elif idx in new_ordering['core']:
            color = 'orange'  # Within core distance
        elif idx in new_ordering['reach']:
            color = 'blue'    # Within max epsilon distance
        else:
            color = 'yellow'  # Unprocessed points
        ax1.scatter(x, y, c=color, edgecolors='black', s=600)  # Increased size
        ax1.text(x, y, labels[idx], fontsize=12, ha='center', va='center')

    # Draw circles for core distance and max_eps
    if current_idx is not None:
        # Core distance
        core_d = core_distances[current_idx]
        if core_d != np.inf:
            circle_core = Circle((points[current_idx][0], points[current_idx][1]), core_d, 
                                 edgecolor='black', linestyle='dotted', fill=False, linewidth=2)
            ax1.add_patch(circle_core)
        # Max epsilon distance
        circle_eps = Circle((points[current_idx][0], points[current_idx][1]), max_eps, 
                            edgecolor='red', linestyle='dotted', fill=False, linewidth=2)
        ax1.add_patch(circle_eps)

    # Add the legend
    create_legend(ax1)
    
    # Add the inset for the reachability plot
    axins = inset_axes(ax1, width="30%", height="30%", loc='lower left', borderpad=5)
    axins.set_title("Reachability Plot", fontsize=12)
    axins.set_xlabel("Order", fontsize=12)
    axins.set_ylabel("Reach Dist", fontsize=12)
    axins.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot the reachability plot as a bar plot with default color (black)
    axins.bar(range(1, len(reach_plot_y)+1), reach_plot_y, color='black', edgecolor='black')
    
    # Set x-ticks and labels
    axins.set_xticks(range(1, len(reach_plot_y)+1))
    axins.set_xticklabels([labels[idx] for idx in ordering], rotation=90, fontsize=12)
    
    # Add the seed list text box
    seen_labels = set()
    filtered_seed_list = []
    for label, dist in seed_list:
        if label not in seen_labels:
            filtered_seed_list.append((label, dist))
            seen_labels.add(label)

    # Now create the seed_text from the filtered list
    seed_text = "Seed list: " + ", ".join([f"({label}, {dist:.1f})" for label, dist in filtered_seed_list])

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax1.text(0.05, 0.95, seed_text, transform=ax1.transAxes, fontsize=15,
             verticalalignment='top', bbox=props)
    
    # Save the updated plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"step_{step}.png"))
    plt.close()

# Function to create the final clustering plot
def create_final_plot(points, labels, reachability, ordering, clusters, cluster_colors, output_dir):
    """
    Create and save the final OPTICS clustering visualization.

    Parameters:
    - points: Numpy array of point coordinates.
    - labels: List of point labels.
    - reachability: List of reachability distances.
    - ordering: List of point indices in processing order.
    - clusters: List of cluster labels for each point.
    - cluster_colors: Dictionary mapping cluster labels to colors.
    - output_dir: Directory to save the final plot.
    """
    # Map each point to its cluster color
    point_colors = [cluster_colors[clusters[idx]] for idx in range(len(points))]
    
    # Create the main plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title("OPTICS Final Clustering")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 50)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_xlabel("X", fontsize=15)
    ax1.set_ylabel("Y", fontsize=15)

    # Set the tick label size
    ax1.tick_params(axis='both', labelsize=15)

    
    # Plot points colored by cluster
    for idx, (x, y) in enumerate(points):
        color = point_colors[idx]
        ax1.scatter(x, y, c=[color], edgecolors='black', s=600)
        ax1.text(x, y, labels[idx], fontsize=12, ha='center', va='center')
    
    # Add the legend for clusters
    cluster_patches = [Patch(facecolor=cluster_colors[cluster], edgecolor='black', label=f'Cluster {cluster}') for cluster in np.unique(clusters)]
    
    # Define legend elements for distance circles using Line2D
    distance_elements = [
        Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='$\epsilon_{core}$'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='$\epsilon_{max}$')
    ]
    
    # Combine both lists
    legend_elements = cluster_patches + distance_elements
    
    # Add the legend to the axis
    ax1.legend(handles=legend_elements, loc='upper right', prop={'size': 15})
    
    # Add the inset for the reachability plot
    axins = inset_axes(ax1, width="30%", height="30%", loc='lower left', borderpad=5)
    axins.set_title("Reachability Plot", fontsize=12)
    axins.set_xlabel("Order", fontsize=12)
    axins.set_ylabel("Reach Dist", fontsize=12)
    axins.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot the reachability plot as a bar plot with colors matching clusters
    for i, idx in enumerate(ordering):
        cluster = clusters[idx]
        color = cluster_colors[cluster]
        axins.bar(i+1, reachability[idx], color=color, edgecolor='black')
    
    # Set x-ticks and labels
    axins.set_xticks(range(1, len(ordering)+1))
    axins.set_xticklabels([labels[idx] for idx in ordering], rotation=90, fontsize=12)
    
    # Save the final clustering plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"final_clustering.png"))
    plt.close()

# Function to create an animated GIF from the saved steps
def create_gif(output_dir, gif_name='optics_animation.gif', duration=1):
    images = []
    # Sort the files based on step number
    step_files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith('step_') and f.endswith('.png')],
        key=lambda x: int(x.split('_')[1].split('.png')[0])
    )
    for step_file in step_files:
        if step_file.endswith('.png'):
            images.append(imageio.imread(os.path.join(output_dir, step_file)))
    imageio.mimsave(gif_name, images, duration=duration)
    print(f"Animated GIF saved as {gif_name}")

# Function to perform step-by-step OPTICS
def run_optics():
    step = 1  # Initialize step counter for saved steps
    for i in range(N):
        if not processed[i]:
            # Start Processing Point i
            current_idx = i
            ordering.append(current_idx)
            reachability[current_idx] = np.inf  # First point has infinite reachability
            reach_plot_y.append(reachability[current_idx])
            neighbors = get_neighbors(current_idx, points, max_eps)
            core_dist = core_distance(points[current_idx], neighbors, min_samples)
            core_distances[current_idx] = core_dist

            if core_dist != np.inf:
                # Initialize priority queue
                queue = PriorityQueue()
                for neighbor in neighbors:
                    neighbor_idx = np.where((points == neighbor).all(axis=1))[0][0]
                    if not processed[neighbor_idx]:
                        new_reach = max(core_dist, euclidean(points[current_idx], points[neighbor_idx]))
                        if reachability[neighbor_idx] is None:
                            reachability[neighbor_idx] = new_reach
                            queue.add_point(neighbor_idx, new_reach)
                        else:
                            if new_reach < reachability[neighbor_idx]:
                                reachability[neighbor_idx] = new_reach
                                queue.add_point(neighbor_idx, new_reach)

                # Determine core and reach points for visualization
                core_neighbors = [idx for idx in range(N) if reachability[idx] is not None and not processed[idx]]
                reach_neighbors = [idx for idx in range(N) if reachability[idx] is not None and not processed[idx] and idx not in core_neighbors]

                # Get the current seed list
                seed_list = queue.get_seed_list(processed)

                # Save Step After Identifying Core Points
                update_plot(step, current_idx, {'core': core_neighbors, 'reach': reach_neighbors}, reach_plot_y, seed_list)
                print(f"Saved Step {step}: Processing Point {labels[current_idx]}")
                step += 1

                while not queue.is_empty():
                    next_idx = queue.pop_point(processed)
                    if next_idx is None:
                        break
                    if not processed[next_idx]:
                        # Start Processing Next Point
                        current_idx = next_idx
                        ordering.append(next_idx)
                        reach_plot_y.append(reachability[next_idx])

                        neighbors_next = get_neighbors(current_idx, points, max_eps)
                        core_dist_next = core_distance(points[current_idx], neighbors_next, min_samples)
                        core_distances[next_idx] = core_dist_next

                        # Update reachability distances
                        if core_dist_next != np.inf:
                            for neighbor in neighbors_next:
                                neighbor_idx = np.where((points == neighbor).all(axis=1))[0][0]
                                if not processed[neighbor_idx]:
                                    new_reach = max(core_dist_next, euclidean(points[current_idx], points[neighbor_idx]))
                                    if reachability[neighbor_idx] is None:
                                        reachability[neighbor_idx] = new_reach
                                        queue.add_point(neighbor_idx, new_reach)
                                    else:
                                        if new_reach < reachability[neighbor_idx]:
                                            reachability[neighbor_idx] = new_reach
                                            queue.add_point(neighbor_idx, new_reach)

                        # Determine core and reach points for visualization
                        core = []
                        reach = []
                        if core_dist_next != np.inf:
                            for neighbor in neighbors_next:
                                neighbor_idx = np.where((points == neighbor).all(axis=1))[0][0]
                                if reachability[neighbor_idx] <= core_dist_next:
                                    core.append(neighbor_idx)
                                elif reachability[neighbor_idx] <= max_eps:
                                    reach.append(neighbor_idx)

                        # Get the current seed list
                        seed_list = queue.get_seed_list(processed)

                        # Save Step After Identifying Core Points
                        update_plot(step, current_idx, {'core': core, 'reach': reach}, reach_plot_y, seed_list)
                        print(f"Saved Step {step}: Processing Point {labels[current_idx]}")
                        step += 1

                        # Mark the Point as Processed After Visualization
                        processed[next_idx] = True

            # Mark the Initial Point as Processed After Visualization
            processed[i] = True

# Function to create the additional figure as per user request
def create_core_max_eps_plot_point_M(points, labels, core_distances, max_eps, output_dir):
    """
    Create and save an additional figure showing all points in yellow,
    highlighting points M, N, P in orange and H, G in blue, and marking
    the core distance and max epsilon distance with circles and half diameter lines for point M only.

    Parameters:
    - points: Numpy array of point coordinates.
    - labels: List of point labels.
    - core_distances: List of core distances for each point.
    - max_eps: Maximum epsilon distance.
    - output_dir: Directory to save the additional plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("OPTICS Core and Max Epsilon Distances for Point M")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 80)
    ax.set_xlabel("X [-]")
    ax.set_ylabel("Y [-]")
    ax.set_xlabel("X", fontsize=15)
    ax.set_ylabel("Y", fontsize=15)

    # Set the tick label size
    ax.tick_params(axis='both', labelsize=15)

    
    
    # Plot all points as yellow
    for idx, (x, y) in enumerate(points):
        ax.scatter(x, y, c='yellow', edgecolors='black', s=400)
        ax.text(x, y, labels[idx], fontsize=12, ha='center', va='center')
    
    # Highlight points M, N, P in orange and H, G in blue
    orange_points = ['M', 'O', 'P']
    blue_points = ['H', 'G','N']
    for idx, label in enumerate(labels):
        if label in orange_points:
            ax.scatter(points[idx][0], points[idx][1], c='orange', edgecolors='black', s=400)
        elif label in blue_points:
            ax.scatter(points[idx][0], points[idx][1], c='blue', edgecolors='black', s=400)
    
    # Identify index for point M
    point_M_label = 'M'
    try:
        point_M_idx = labels.index(point_M_label)
    except ValueError:
        print(f"Label {point_M_label} not found in labels.")
        return
    
    # Get coordinates for point M
    x_M, y_M = points[point_M_idx]
    core_d_M = core_distances[point_M_idx]
    
    # Draw the core distance circle for point M
    if core_d_M != np.inf:
        circle_core = Circle((x_M, y_M), core_d_M, edgecolor='black', linestyle='dotted', fill=False, linewidth=2)
        ax.add_patch(circle_core)
    
    # Draw the max epsilon distance circle for point M
    circle_eps = Circle((x_M, y_M), max_eps, edgecolor='red', linestyle='dotted', fill=False, linewidth=2)
    ax.add_patch(circle_eps)
    
    # Draw half diameter lines (radius lines) for point M at different angles
    if core_d_M != np.inf:
        # Define angles in degrees
        angle_core = 45  # Degrees for epsilon_core
        angle_max = -45  # Degrees for epsilon_max
        
        # Convert angles to radians
        theta_core = math.radians(angle_core)
        theta_max = math.radians(angle_max)
        
        # Calculate end points for epsilon_core
        x_core_end = x_M + core_d_M * math.cos(theta_core)
        y_core_end = y_M + core_d_M * math.sin(theta_core)
        
        # Calculate end points for epsilon_max
        x_max_end = x_M + max_eps * math.cos(theta_max)
        y_max_end = y_M + max_eps * math.sin(theta_max)
        
        # Draw epsilon_core line
        ax.plot([x_M, x_core_end], [y_M, y_core_end], linestyle='--', color='black', linewidth=2)
        # Annotate epsilon_core
        ax.text(x_core_end-1, y_core_end + 2, r'$\epsilon_{core}$', color='black', fontsize=20, ha='center')
        
        # Draw epsilon_max line
        ax.plot([x_M, x_max_end], [y_M, y_max_end], linestyle='--', color='red', linewidth=2)
        # Annotate epsilon_max
        ax.text(x_max_end+1, y_max_end + 2, r'$\epsilon_{max}$', color='red', fontsize=20, ha='center')
    
    # Create a simplified legend
    legend_elements = [
        Patch(facecolor='yellow', edgecolor='black', label='all Points'),
        Patch(facecolor='orange', edgecolor='black', label='core point ($\in$ seed points)'),
        Patch(facecolor='blue', edgecolor='black', label='seed point')
    ]
    ax.legend(handles=legend_elements, loc='upper right', prop={'size': 15})
    
    # Save the additional figure
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(output_dir, f"core_max_eps_point_M.png"))
    plt.close()

# Main execution flow
if __name__ == "__main__":
    # Initialize the first plot
    initialize_plot()
    
    # Run the OPTICS algorithm with visualization
    run_optics()
    
    # Assign clusters based on reachability distances
    clusters = assign_clusters(points, reachability, min_samples, max_eps)
    
    # Map clusters to colors
    cluster_colors = map_clusters_to_colors(clusters)
    
    # Create and save the final clustering plot
    create_final_plot(points, labels, reachability, ordering, clusters, cluster_colors, output_dir)
    
    # Create the additional core and max epsilon distance plot for point M
    create_core_max_eps_plot_point_M(points, labels, core_distances, max_eps, output_dir)
    
    # Create an animated GIF (optional)
    create_gif(output_dir)
    
    print(f"OPTICS step-by-step visualization completed. Check the '{output_dir}' directory for images and the animated GIF.")
