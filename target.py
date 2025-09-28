import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # Add this import
import seaborn as sns
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
from scipy.spatial import ConvexHull

# Add this function to your file after the imports but before the other functions
def standardize_target_image(image, center_point, crop_size=1600):
    """
    Standardize target image by cropping to a fixed size around the detected center.
    
    Args:
        image: OpenCV image
        center_point: (x, y) coordinates of the target center
        crop_size: Size of the output square image in pixels
    
    Returns:
        Standardized image cropped around center, new center coordinates
    """
    h, w = image.shape[:2]
    # Use 1/8 of crop_size instead of half (1/2) for a much smaller crop
    crop_radius = crop_size // 2
    
    # Calculate crop boundaries with center point in the middle
    x_center, y_center = center_point
    
    left = max(x_center - crop_radius, 0)
    top = max(y_center - crop_radius, 0)
    right = min(x_center + crop_radius, w)
    bottom = min(y_center + crop_radius, h)
    
    # Handle edge cases where the crop would go beyond image boundaries
    if right - left < crop_radius * 2:
        if left == 0:
            right = min(crop_radius * 2, w)
        elif right == w:
            left = max(w - crop_radius * 2, 0)
    
    if bottom - top < crop_radius * 2:
        if top == 0:
            bottom = min(crop_radius * 2, h)
        elif bottom == h:
            top = max(h - crop_radius * 2, 0)
    
    # Crop the image
    cropped = image[int(top):int(bottom), int(left):int(right)]
    
    # If the cropped image is smaller than the intended size,
    # create a black canvas and place the image in the center
    ch, cw = cropped.shape[:2]
    if ch < crop_radius * 2 or cw < crop_radius * 2:
        canvas = np.zeros((crop_radius * 2, crop_radius * 2, 3), dtype=np.uint8)
        x_offset = (crop_radius * 2 - cw) // 2
        y_offset = (crop_radius * 2 - ch) // 2
        canvas[y_offset:y_offset+ch, x_offset:x_offset+cw] = cropped
        cropped = canvas
        
        # Adjust center point for the new canvas
        new_center = (
            x_center - left + x_offset,
            y_center - top + y_offset
        )
    else:
        # If we need to resize to exact dimensions
        if cropped.shape[0] != crop_radius * 2 or cropped.shape[1] != crop_radius * 2:
            cropped = cv2.resize(cropped, (crop_radius * 2, crop_radius * 2))
            
        # Calculate new center coordinates in the cropped image
        new_center = (
            x_center - left,
            y_center - top
        )
    
    return cropped, new_center

# ----------------- Scoring Based on Your Formula -------------------
def calculate_score(distance_cm):
    if distance_cm <= 0.25:
        return "X"
    elif distance_cm <= 0.5:
        return 10
    elif distance_cm <= 1.4:
        return 9
    elif distance_cm <= 2.0:
        return 8
    elif distance_cm <= 3.3:
        return 7
    elif distance_cm <= 4.0:
        return 6
    elif distance_cm <= 4.7:
        return 5
    elif distance_cm <= 5.5:
        return 4
    elif distance_cm <= 6.3:
        return 3
    elif distance_cm <= 7.1:
        return 2
    elif distance_cm <= 7.8:
        return 1
    else:
        return 0

# ----------------- Helper: Euclidean Distance -------------------
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ----------------- Shot Group Analysis Functions -------------------
def calculate_group_statistics(shot_points, center_point, distances_cm):
    """Calculate statistics about the shot group"""
    stats = {}
    
    # Convert X scores to numeric 10.5 for calculations
    numeric_scores = []
    for score in scores:
        numeric_scores.append(10.5 if score == "X" else float(score))
    
    # Basic stats
    stats['total_shots'] = len(shot_points)
    stats['average_score'] = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0
    stats['min_distance'] = min(distances_cm) if distances_cm else 0
    stats['max_distance'] = max(distances_cm) if distances_cm else 0
    stats['mean_distance'] = sum(distances_cm) / len(distances_cm) if distances_cm else 0
    stats['std_deviation'] = np.std(distances_cm) if distances_cm else 0
    
    # Mean point of impact (MPI)
    if shot_points:
        mpi_x = sum(p[0] for p in shot_points) / len(shot_points)
        mpi_y = sum(p[1] for p in shot_points) / len(shot_points)
        stats['mpi'] = (mpi_x, mpi_y)
        
        # MPI distance from target center
        if center_point:
            mpi_distance_px = euclidean_distance((mpi_x, mpi_y), center_point)
            # Convert to cm using the same ratio as other measurements
            if distances_cm and shot_points:
                px_per_cm = euclidean_distance(shot_points[0], center_point) / (distances_cm[0] or 1)
                stats['mpi_distance_cm'] = mpi_distance_px / px_per_cm if px_per_cm else 0
    
    # Extreme spread (maximum distance between any two shots)
    if len(shot_points) >= 2:
        max_spread = 0
        for i, p1 in enumerate(shot_points):
            for p2 in shot_points[i+1:]:
                spread = euclidean_distance(p1, p2)
                if spread > max_spread:
                    max_spread = spread
        
        if distances_cm and shot_points:
            px_per_cm = euclidean_distance(shot_points[0], center_point) / (distances_cm[0] or 1)
            stats['extreme_spread_cm'] = max_spread / px_per_cm if px_per_cm else 0
    
    # Calculate group size (diameter of smallest circle containing all shots)
    if len(shot_points) >= 3:
        try:
            hull = ConvexHull(shot_points)
            hull_points = [shot_points[i] for i in hull.vertices]
            hull_diameter = 0
            for i, p1 in enumerate(hull_points):
                for p2 in hull_points[i+1:]:
                    diameter = euclidean_distance(p1, p2)
                    if diameter > hull_diameter:
                        hull_diameter = diameter
            
            if distances_cm and shot_points:
                px_per_cm = euclidean_distance(shot_points[0], center_point) / (distances_cm[0] or 1)
                stats['group_diameter_cm'] = hull_diameter / px_per_cm if px_per_cm else 0
        except:
            # If ConvexHull fails, fall back to extreme spread
            stats['group_diameter_cm'] = stats.get('extreme_spread_cm', 0)
    
    return stats

# ----------------- Main Inference + Scoring Function -------------------
def process_yolo_with_score(image_path, model_path, standardize=True):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(model_path)
    results = model.predict(image)[0]

    centers = []
    shots = []
    radius = None

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if cls == 0:
            centers.append((cx, cy))
            radius = (x2 - x1) / 2
        
        elif cls == 1:
            shots.append(((cx, cy), (x1, y1, x2, y2)))

    # Skip if no center detected
    if not centers:
        return image, [], None, [], []
        
    # Use the first center detected
    center = centers[0]
    
    # Standardize the image if requested
    if standardize:
        # Standardize image with the detected center
        std_image, new_center = standardize_target_image(image, center)
        
        # Recalculate shot positions in the standardized image
        std_shots = []
        for (sx, sy), bbox in shots:
            # Calculate the offset from original center to shot
            dx = sx - center[0]
            dy = sy - center[1]
            
            # Apply the same offset from new center
            new_sx = new_center[0] + dx
            new_sy = new_center[1] + dy
            
            # Add to standardized shots
            std_shots.append(((new_sx, new_sy), bbox))  # Keep original bbox for now
            
        image = std_image
        center = new_center
        shots = std_shots
    
    # Now proceed with your existing code for adding annotations
    shot_coords = [pt for pt, _ in shots]
    shot_scores = []
    distances_list = []

    for shot_center, bbox in shots:
        distance_px = euclidean_distance(shot_center, center)
        distance_cm = (distance_px * 0.25) / radius if radius else 0
        score = calculate_score(distance_cm)
        distances_list.append(distance_cm)
        shot_scores.append(score)

        # Draw annotations on the image
        cv2.circle(image, (int(center[0]), int(center[1])), 6, (255, 0, 255), -1)
        cv2.putText(image, "Center", (int(center[0]) - 30, int(center[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        cv2.circle(image, (int(shot_center[0]), int(shot_center[1])), 6, (255, 255, 255), -1)
        cv2.putText(image, "Shot", (int(shot_center[0]) - 25, int(shot_center[1]) + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.line(image, 
                (int(shot_center[0]), int(shot_center[1])), 
                (int(center[0]), int(center[1])), 
                (255, 255, 0), 2)
                
        mid_x = (shot_center[0] + center[0]) // 2
        mid_y = (shot_center[1] + center[1]) // 2

        cv2.putText(image, f"{distance_cm:.2f} cm", (int(mid_x) - 40, int(mid_y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(image, f"Score: {score}", (int(shot_center[0]) + 10, int(shot_center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return image, shot_coords, center, distances_list, shot_scores

# ----------------- Enhanced Plotting Functions -------------------
def plot_shots_on_target(image_path, shot_points, center_point, distances_cm, scores, stats=None, save_path=None):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_rgb)

    # Draw scoring rings
    ring_radii_cm = [0.5, 0.7, 1.25, 2.1, 2.5, 3.5, 4.0, 4.5, 5.3, 5.5, 6.0]
    if center_point and distances_cm and distances_cm[0] > 0:
        px_per_cm = euclidean_distance(shot_points[0], center_point) / distances_cm[0]
        for r in ring_radii_cm:
            radius_px = r * px_per_cm
            circle = plt.Circle(center_point, radius_px, color='white', fill=False, linestyle='--', alpha=0.3)
            ax.add_patch(circle)

    # Draw heat map for shot distribution
    if len(shot_points) >= 2:
        x_vals = [pt[0] for pt in shot_points]
        y_vals = [pt[1] for pt in shot_points]
        sns.kdeplot(x=x_vals, y=y_vals, cmap="Reds", fill=True, bw_adjust=0.5, alpha=0.6, ax=ax)

    # Mark center and shots
    if center_point:
        ax.plot(center_point[0], center_point[1], "mo", markersize=10, label="Center")

    for i, (pt, score) in enumerate(zip(shot_points, scores)):
        ax.plot(pt[0], pt[1], "wo", markersize=8)
        ax.text(pt[0] + 5, pt[1], f"{i+1}: {score}", color="yellow", fontsize=9, weight='bold')
    
    # Draw Mean Point of Impact if available
    if stats and 'mpi' in stats:
        mpi = stats['mpi']
        ax.plot(mpi[0], mpi[1], "gx", markersize=12, label="MPI")
        # Draw circle around group
        if 'group_diameter_cm' in stats and center_point and distances_cm:
            px_per_cm = euclidean_distance(shot_points[0], center_point) / distances_cm[0] if distances_cm[0] > 0 else 40
            group_radius_px = (stats['group_diameter_cm'] / 2) * px_per_cm
            group_circle = plt.Circle(mpi, group_radius_px, color='lime', fill=False, linestyle='-', alpha=0.7)
            ax.add_patch(group_circle)

    # Add statistics text box if available
    if stats:
        stat_text = (
            f"Total shots: {stats['total_shots']}\n"
            f"Avg score: {stats['average_score']:.2f}\n"
            f"Mean radius: {stats['mean_distance']:.2f} cm\n"
            f"Group size: {stats.get('group_diameter_cm', 0):.2f} cm\n"
            f"MPI offset: {stats.get('mpi_distance_cm', 0):.2f} cm"
        )
        props = dict(boxstyle='round', facecolor='black', alpha=0.7)
        ax.text(0.05, 0.95, stat_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', color='white', bbox=props)

    ax.set_title("Professional Shot Analysis", fontsize=14)
    ax.legend(loc='upper right')
    ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to: {save_path}")
    plt.close()

def plot_score_map(shot_points, center_point, distances_cm, scores, stats=None, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Define the ring radii (in cm) and corresponding labels
    ring_radii_cm = [0.5, 0.7, 1.25, 2.1, 2.5, 3.5, 4.0, 4.5, 5.5, 6.5]
    ring_labels = ["X", "10", "9", "8", "7", "6", "5", "4", "3", "2"]
    
    # Use the first distance to estimate pixels per cm
    if center_point and distances_cm and distances_cm[0] > 0:
        px_per_cm = euclidean_distance(shot_points[0], center_point) / distances_cm[0]
    else:
        px_per_cm = 40  # fallback value
    
    # Draw scoring rings with labels
    for r, label in zip(ring_radii_cm, ring_labels):
        circle = plt.Circle(center_point, r * px_per_cm, color='white', fill=False, linewidth=1.2)
        ax.add_patch(circle)
        # Add ring labels
        angle = np.pi / 4  # 45 degrees
        label_x = center_point[0] + (r * px_per_cm * np.cos(angle))
        label_y = center_point[1] + (r * px_per_cm * np.sin(angle))
        ax.text(label_x, label_y, label, color='white', fontsize=9, 
                ha='center', va='center', weight='bold')
    
    # Create a color map for scores (higher score = brighter color)
    cmap = plt.get_cmap("coolwarm_r")
    norm_scores = []
    for s in scores:
        norm_scores.append(10.5 if s == "X" else float(s))
    
    # Normalize between 0 and 1 for color mapping
    max_score = 10.5
    min_score = 0
    normalized = [(s - min_score) / (max_score - min_score) for s in norm_scores]

    # Plot each shot with number and score
    for i, ((x, y), score_norm, score) in enumerate(zip(shot_points, normalized, scores)):
        ax.plot(x, y, 'o', color=cmap(score_norm), markersize=12)
        ax.text(x, y, f"{i+1}", color='black', fontsize=8, 
                ha='center', va='center', weight='bold')
    
    # Draw Mean Point of Impact if available
    if stats and 'mpi' in stats:
        mpi = stats['mpi']
        ax.plot(mpi[0], mpi[1], "gx", markersize=12, label="MPI")
    
    # Add group size circle if available
    if stats and 'group_diameter_cm' in stats and center_point and distances_cm:
        mpi = stats['mpi']
        group_radius_px = (stats['group_diameter_cm'] / 2) * px_per_cm
        group_circle = plt.Circle(mpi, group_radius_px, color='lime', fill=False, linestyle='-', alpha=0.7)
        ax.add_patch(group_circle)

    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.axis('off')
    ax.set_title("Professional Scoring Analysis", color='white', fontsize=14)
    
    # Add statistics text box if available
    if stats:
        stat_text = (
            f"Average: {stats['average_score']:.2f}\n"
            f"Dispersion: {stats.get('std_deviation', 0):.2f} cm\n"
            f"Group size: {stats.get('group_diameter_cm', 0):.2f} cm"
        )
        props = dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white')
        ax.text(0.05, 0.05, stat_text, transform=ax.transAxes, fontsize=10,
                color='white', bbox=props)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor='black')
        print(f"Score map saved to: {save_path}")
    plt.close()



def create_professional_consolidated_scoremap(all_shot_positions_cm, all_scores, player_name, output_folder):
    """
    Create a consolidated score map with professional design matching the reference image:
    - Tan/beige background
    - Black center with white concentric rings
    - Colored shots (green/yellow)
    - Score display in corner
    """
    # Create blank tan/beige canvas
    canvas_size = 1800  # Size of the canvas
    center_point = (canvas_size//2, canvas_size//2)  # Center of the canvas
    
    # Create tan/beige background (RGB: 245, 222, 179)
    consolidated_image = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * np.array([179, 222, 245], dtype=np.uint8)
    
    # Set standard cm to pixel ratio
    standard_px_per_cm = 100  # 100 pixels = 1 cm (can adjust as needed)
    
    # Create black center area (like in reference image)
    black_center_radius = int(8.5 * standard_px_per_cm)  # Approx size of black center
    cv2.circle(consolidated_image, center_point, black_center_radius, (0, 0, 0), -1)
    
    # Define scoring rings (in cm)
    # Include more rings than scoring rings to match reference image
    ring_radii_cm = [0.5, 1.0, 1.8, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
    
    # Draw all scoring rings in white
    for radius_cm in ring_radii_cm:
        radius_px = int(radius_cm * standard_px_per_cm)
        cv2.circle(consolidated_image, center_point, radius_px, (255, 255, 255), 1)
    
    # Add numeric ring labels at evenly spaced positions around the circles, like in reference image
    labels = ["X", "10", "9", "8", "7", "6", "5", "4", "3", "2" , "1"]
    scoring_radii_cm = [0.25, 0.5, 1.4, 2.0, 3.3, 4.0, 4.7, 5.5, 6.3, 7.1 ,7.8]  # These correspond to scoring areas
    
    # Place labels around the rings in 4 directions (like in the reference image)
    for radius_cm, label in zip(scoring_radii_cm, labels):
        radius_px = int(radius_cm * standard_px_per_cm)
        
        # Add the label at multiple positions around circle
        for angle_deg in [0, 90, 180, 270]:  # right, top, left, bottom
            angle_rad = math.radians(angle_deg)
            label_x = int(center_point[0] + radius_px * math.cos(angle_rad))
            label_y = int(center_point[1] + radius_px * math.sin(angle_rad))
            
            # Adjust for text placement
            if angle_deg == 0:  # right
                cv2.putText(consolidated_image, label, (label_x+5, label_y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            elif angle_deg == 90:  # top
                cv2.putText(consolidated_image, label, (label_x-5, label_y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            elif angle_deg == 180:  # left
                cv2.putText(consolidated_image, label, (label_x-15, label_y+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            elif angle_deg == 270:  # bottom
                cv2.putText(consolidated_image, label, (label_x-5, label_y+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Calculate score percentage for display (like "98/100" in the reference)
    total_possible = 10.5 * len(all_shot_positions_cm)  # maximum possible score
    actual_score = sum([10.5 if s == "X" else float(s) for s in all_scores])
    score_percentage = int((actual_score / total_possible) * 100)
    
    # Add score text in top right corner like reference image
    score_text = f"{score_percentage}/100"
    cv2.putText(consolidated_image, score_text, (canvas_size-300, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Plot all shots in green/yellow like in reference image
    # Center shot will be yellow, others green
    
    # Find the best shot (closest to center)
    best_shot_idx = all_distances.index(min(all_distances)) if all_distances else -1
    
    for i, (x_cm, y_cm) in enumerate(all_shot_positions_cm):
        # Convert cm position to pixel position 
        x_px = center_point[0] + int(x_cm * standard_px_per_cm)
        y_px = center_point[1] - int(y_cm * standard_px_per_cm)  # Flip y-axis
        
        # Ensure coordinates are within image bounds
        if 0 <= x_px < canvas_size and 0 <= y_px < canvas_size:
            # Best shot is yellow, others green (like in reference image)
            color = (0, 255, 255) if i == best_shot_idx else (0, 255, 0)  # BGR format
            # Draw colored circle for shot
            cv2.circle(consolidated_image, (x_px, y_px), 15, color, -1)
    
    # Save the consolidated image
    consolidated_scoremap_path = os.path.join(output_folder, "plots", f"{player_name}_professional_scoremap.png")
    cv2.imwrite(consolidated_scoremap_path, consolidated_image)
    print(f"Professional consolidated score map saved to: {consolidated_scoremap_path}")
    
    return consolidated_image

def create_consolidated_scoremap_on_real_target(all_shot_positions_cm, all_scores, player_name, output_folder, first_image_path, consolidated_stats=None):
    """
    Create a consolidated score map using the first detected target as the base image.
    Overlay all shots from all targets onto this single real target image.
    """
    # Read the first target image as the base
    base_image = cv2.imread(first_image_path)
    if base_image is None:
        print(f"Could not read base image: {first_image_path}. Using default background.")
        # Fall back to the professional score map if image can't be loaded
        return create_professional_consolidated_scoremap(all_shot_positions_cm, all_scores, player_name, output_folder)
    
    # Process the base image to detect the center
    model = YOLO("best.pt")
    results = model.predict(base_image)[0]
    
    center = None
    radius = None
    
    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        if cls == 0:  # Center class
            center = (cx, cy)
            radius = (x2 - x1) / 2
            break
    
    if not center or not radius:
        print("Could not detect center in base image. Using default background.")
        # Fall back to the professional score map if center can't be detected
        return create_professional_consolidated_scoremap(all_shot_positions_cm, all_scores, player_name, output_folder)
    
    # Standardize the base image to make sure center is properly positioned
    std_image, new_center = standardize_target_image(base_image, center)
    
    # Set standard cm to pixel ratio based on the detected radius
    standard_px_per_cm = radius * 4  # Adjust this multiplier if needed for proper scaling
    
    # Calculate score percentage
    total_possible = 10.5 * len(all_shot_positions_cm)  # maximum possible score
    actual_score = sum([10.5 if s == "X" else float(s) for s in all_scores])
    score_percentage = int((actual_score / total_possible) * 100)
    
    # Add score text in top right corner
    score_text = f"{score_percentage}/100"
    cv2.putText(std_image, score_text, (std_image.shape[1]-300, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)  # Black with thicker outline
    cv2.putText(std_image, score_text, (std_image.shape[1]-300, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)  # White text
    
    # Find the best shot (closest to center)
    best_shot_idx = all_distances.index(min(all_distances)) if all_distances else -1
    
    # Increase shot marker size to match real shot size
    # A real pellet/bullet hole is typically 4.5mm to 5.5mm
    # Convert this to pixels based on our scaling
    shot_size_cm = 0.25  # Typical pellet size in cm
    shot_radius_px = int(shot_size_cm * standard_px_per_cm)
    
    # Make sure shot is visible even with small calibers (minimum size)
    shot_radius_px = max(shot_radius_px, 20)
    
    # Calculate pixel positions for all shots first
    pixel_positions = []
    for x_cm, y_cm in all_shot_positions_cm:
        x_px = new_center[0] + int(x_cm * standard_px_per_cm)
        y_px = new_center[1] - int(y_cm * standard_px_per_cm)  # Flip y-axis
        pixel_positions.append((x_px, y_px))
    
    # Detect overlapping shots (shots that are very close to each other)
    overlap_threshold_px = shot_radius_px * 1.2  # Shots closer than this are considered overlapping
    overlapping_shots = set()
    
    # Check each pair of shots for overlap
    for i in range(len(pixel_positions)):
        for j in range(i+1, len(pixel_positions)):
            x1, y1 = pixel_positions[i]
            x2, y2 = pixel_positions[j]
            distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # If distance is less than threshold, mark both shots as overlapping
            if distance < overlap_threshold_px:
                overlapping_shots.add(i)
                overlapping_shots.add(j)
    
    # Plot all shots on the real target with more realistic sizes
    for i, ((x_cm, y_cm), (x_px, y_px)) in enumerate(zip(all_shot_positions_cm, pixel_positions)):
        # Ensure coordinates are within image bounds
        if 0 <= x_px < std_image.shape[1] and 0 <= y_px < std_image.shape[0]:
            # Use red for overlapping shots, yellow for best shot, green for others
            if i in overlapping_shots:
                color = (0, 0, 255)  # Red (BGR) for overlapping shots
            elif i == best_shot_idx:
                color = (0, 255, 255)  # Yellow for best shot
            else:
                color = (0, 255, 0)  # Green for normal shots
            
            # Draw colored circle for shot with larger size
            cv2.circle(std_image, (x_px, y_px), shot_radius_px+2, (0, 0, 0), -1)  # Black outline
            cv2.circle(std_image, (x_px, y_px), shot_radius_px, color, -1)  # Colored center
            
            # Add shot number for reference
            cv2.putText(std_image, f"{i+1}", (x_px-5, y_px+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
            cv2.putText(std_image, f"{i+1}", (x_px-5, y_px+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text
    
    # Calculate and display MPI
    if all_shot_positions_cm:
        mpi_x_cm = sum(p[0] for p in all_shot_positions_cm) / len(all_shot_positions_cm)
        mpi_y_cm = sum(p[1] for p in all_shot_positions_cm) / len(all_shot_positions_cm)
        
        # Convert MPI from cm to pixels
        mpi_x_px = new_center[0] + int(mpi_x_cm * standard_px_per_cm)
        mpi_y_px = new_center[1] - int(mpi_y_cm * standard_px_per_cm)  # Flip y-axis
        
        # Draw MPI marker
        cv2.drawMarker(std_image, (mpi_x_px, mpi_y_px), (0, 255, 0), cv2.MARKER_CROSS, 20, 3)
        
        # Add circle showing group size based on standard deviation
        if consolidated_stats and 'std_deviation' in consolidated_stats and consolidated_stats['std_deviation'] > 0:
            group_radius_px = int(consolidated_stats['std_deviation'] * standard_px_per_cm)
            cv2.circle(std_image, (mpi_x_px, mpi_y_px), group_radius_px, (0, 255, 0), 2)
    
    # Add legend for shot colors
    legend_y_start = 200
    legend_x = std_image.shape[1] - 300
    
    # Best shot legend
    cv2.circle(std_image, (legend_x, legend_y_start), shot_radius_px, (0, 255, 255), -1)
    cv2.putText(std_image, "Best shot", (legend_x + 30, legend_y_start + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(std_image, "Best shot", (legend_x + 30, legend_y_start + 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Normal shot legend
    cv2.circle(std_image, (legend_x, legend_y_start + 40), shot_radius_px, (0, 255, 0), -1)
    cv2.putText(std_image, "Normal shot", (legend_x + 30, legend_y_start + 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(std_image, "Normal shot", (legend_x + 30, legend_y_start + 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Overlapping shot legend
    cv2.circle(std_image, (legend_x, legend_y_start + 80), shot_radius_px, (0, 0, 255), -1)
    cv2.putText(std_image, "Overlapping shots", (legend_x + 30, legend_y_start + 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(std_image, "Overlapping shots", (legend_x + 30, legend_y_start + 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Add count of overlapping shots
    if overlapping_shots:
        overlap_count_text = f"Overlapping shots: {len(overlapping_shots)}"
        cv2.putText(std_image, overlap_count_text, (legend_x, legend_y_start + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(std_image, overlap_count_text, (legend_x, legend_y_start + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Save the consolidated image
    consolidated_scoremap_path = os.path.join(output_folder, "plots", f"{player_name}_real_target_consolidated.png")
    cv2.imwrite(consolidated_scoremap_path, std_image)
    print(f"Real target consolidated score map saved to: {consolidated_scoremap_path}")
    
    return std_image

# ----------------- Generate Summary Report -------------------
def generate_summary_report(image_name, shot_points, center_point, distances_cm, scores, stats, save_path=None):
    """Generate a detailed summary report for the shooting session"""
    # Convert X scores to numeric for calculations
    numeric_scores = []
    for score in scores:
        numeric_scores.append(10.5 if score == "X" else float(score))
    
    # Count each score type
    score_counts = {}
    for s in scores:
        score_counts[s] = score_counts.get(s, 0) + 1
    
    # Format the report
    report = (
        f"# Professional Shooting Analysis Report\n"
        f"## Session: {image_name}\n"
        f"## Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"### Overall Statistics\n"
        f"- Total shots: {stats['total_shots']}\n"
        f"- Average score: {stats['average_score']:.2f}\n"
        f"- Total points: {sum(numeric_scores):.1f}\n"
        f"- Minimum distance: {stats['min_distance']:.2f} cm\n"
        f"- Maximum distance: {stats['max_distance']:.2f} cm\n"
        f"- Mean distance: {stats['mean_distance']:.2f} cm\n"
        f"- Standard deviation: {stats['std_deviation']:.2f} cm\n\n"
        f"### Group Analysis\n"
    )
    
    # Handle potential missing stats with safer formatting
    group_diameter = stats.get('group_diameter_cm')
    if group_diameter is not None:
        report += f"- Group diameter: {group_diameter:.2f} cm\n"
    else:
        report += f"- Group diameter: N/A\n"
        
    extreme_spread = stats.get('extreme_spread_cm')
    if extreme_spread is not None:
        report += f"- Extreme spread: {extreme_spread:.2f} cm\n"
    else:
        report += f"- Extreme spread: N/A\n"
        
    mpi_distance = stats.get('mpi_distance_cm')
    if mpi_distance is not None:
        report += f"- MPI distance from center: {mpi_distance:.2f} cm\n\n"
    else:
        report += f"- MPI distance from center: N/A\n\n"
    
    report += f"### Score Distribution\n"
    
    # Add score distribution
    for score in sorted(score_counts.keys(), key=lambda x: 10.5 if x == "X" else float(x), reverse=True):
        count = score_counts[score]
        percentage = (count / len(scores)) * 100
        report += f"- {score}: {count} shots ({percentage:.1f}%)\n"
    
    # Add shot details
    report += "\n### Individual Shot Details\n"
    report += "| Shot # | Score | Distance (cm) |\n"
    report += "|--------|-------|---------------|\n"
    
    for i, (score, dist) in enumerate(zip(scores, distances_cm)):
        report += f"| {i+1} | {score} | {dist:.2f} |\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {save_path}")
    
    return report

# ----------------- Export to CSV -------------------
def export_to_csv(image_name, shot_points, distances_cm, scores, stats, save_path=None):
    """Export shot data to CSV for further analysis"""
    data = []
    
    for i, ((x, y), dist, score) in enumerate(zip(shot_points, distances_cm, scores)):
        shot_data = {
            'session': image_name,
            'shot_number': i + 1,
            'x_coordinate': x,
            'y_coordinate': y,
            'distance_cm': dist,
            'score': score,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        data.append(shot_data)
    
    df = pd.DataFrame(data)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"CSV data saved to: {save_path}")
    
    return df

# ----------------- Main Script -------------------
if __name__ == "__main__":
    # Use the test folder name as the player/shooter name
    player_name = os.path.basename(os.path.normpath("user_images\Abdalrhman"))
    model_path = "best.pt"
    output_folder = "scored_outputs"
    os.makedirs(output_folder, exist_ok=True)
    
    # Create folders for different output types
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "reports"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "data"), exist_ok=True)

    # Collect results across all targets
    all_shot_points = []
    all_centers = []
    all_distances = []
    all_scores = []
    all_image_paths = []
    all_shot_source_images = []  # Track which image each shot came from
    all_shot_positions_cm = []  # Track shot positions in cm
    
    # In the processing loop, store the real-world positions in cm in addition to pixel coordinates
    for filename in os.listdir("user_images\Abdalrhman"):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join("user_images\Abdalrhman", filename)
            print(f"Processing: {image_path}")
            
            # Process image and get basic data
            output_img, shot_pts, center, dists_cm, scores = process_yolo_with_score(image_path, model_path)
            
            # Skip if no shots detected
            if not shot_pts or not center:
                print(f"No shots or center detected in {filename}. Skipping.")
                continue
            
            # Save annotated image
            output_img_path = os.path.join(output_folder, "images", f"scored_{filename}")
            cv2.imwrite(output_img_path, output_img)
            
            # Calculate positions in cm for proper normalization
            shot_positions_cm = []
            for i, shot_pt in enumerate(shot_pts):
                # Calculate vector from center to shot
                dx = shot_pt[0] - center[0]
                dy = center[1] - shot_pt[1]  # Invert y coordinate (y grows downward in images)
                
                # Convert to cm using the distance
                if dists_cm[i] > 0:
                    # Calculate angle
                    angle = math.atan2(dy, dx)
                    # Create position in cm
                    x_cm = math.cos(angle) * dists_cm[i]
                    y_cm = math.sin(angle) * dists_cm[i]
                    shot_positions_cm.append((x_cm, y_cm))
                else:
                    shot_positions_cm.append((0, 0))  # Fallback
            
            # Individual target analysis
            stats = calculate_group_statistics(shot_pts, center, dists_cm)
            
            # Individual target report
            target_num = os.path.splitext(filename)[0]
            report_path = os.path.join(output_folder, "reports", f"target_{target_num}.md")
            generate_summary_report(f"{player_name} - Target {target_num}", 
                                  shot_pts, center, dists_cm, scores, stats, 
                                  save_path=report_path)
            
            # Add to consolidated analysis
            all_shot_points.extend(shot_pts)
            all_centers.append(center)
            all_distances.extend(dists_cm)
            all_scores.extend(scores)
            all_image_paths.append(image_path)
            all_shot_source_images.extend([filename] * len(shot_pts))
            
            # Store positions in cm for proper consolidation
            if 'shot_positions_cm' not in locals():
                shot_positions_cm = []
            all_shot_positions_cm.extend(shot_positions_cm)
    
    # Skip consolidated analysis if no shots detected
    if not all_shot_points:
        print("No shots detected in any images. Exiting.")
        exit()

    # Get the first image path for the real target base
    first_image_path = os.path.join("user_images\Abdalrhman", all_shot_source_images[0]) if all_shot_source_images else None

    # Create consolidated score map on real target image (if available)
    if first_image_path and os.path.exists(first_image_path):
        create_consolidated_scoremap_on_real_target(
            all_shot_positions_cm, 
            all_scores, 
            player_name, 
            output_folder,
            first_image_path
        )
    
    # Create consolidated CSV data
    consolidated_csv_path = os.path.join(output_folder, "data", f"{player_name}_consolidated.csv")
    consolidated_data = []
    
    for i, ((x, y), dist, score, source_img) in enumerate(
            zip(all_shot_points, all_distances, all_scores, all_shot_source_images)):
        shot_data = {
            'player': player_name,
            'shot_number': i + 1,
            'x_coordinate': x,
            'y_coordinate': y,
            'distance_cm': dist,
            'score': score,
            'source_image': source_img,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        consolidated_data.append(shot_data)
    
    df = pd.DataFrame(consolidated_data)
    df.to_csv(consolidated_csv_path, index=False)
    print(f"Consolidated CSV data saved to: {consolidated_csv_path}")
    
    # Create consolidated statistics
    consolidated_stats = {}
    
    # Convert X scores to numeric 10.5 for calculations
    numeric_scores = []
    for score in all_scores:
        numeric_scores.append(10.5 if score == "X" else float(score))
    
    # Basic consolidated stats
    consolidated_stats['total_shots'] = len(all_shot_points)
    consolidated_stats['average_score'] = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0
    consolidated_stats['min_distance'] = min(all_distances) if all_distances else 0
    consolidated_stats['max_distance'] = max(all_distances) if all_distances else 0
    consolidated_stats['mean_distance'] = sum(all_distances) / len(all_distances) if all_distances else 0
    consolidated_stats['std_deviation'] = np.std(all_distances) if all_distances else 0
    
    # Create special consolidated report
    consolidated_report_path = os.path.join(output_folder, f"{player_name}_performance_report.md")
    
    report = (
        f"# Professional Shooting Analysis: {player_name}\n"
        f"## Session Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
        f"### Overall Performance\n"
        f"- Total targets: {len(set(all_shot_source_images))}\n"
        f"- Total shots: {consolidated_stats['total_shots']}\n"
        f"- Average score: {consolidated_stats['average_score']:.2f}\n"
        f"- Total points: {sum(numeric_scores):.1f}\n"
        f"- Score percentage: {(sum(numeric_scores)/(consolidated_stats['total_shots']*10.5))*100:.2f}%\n\n"
    )
    
    # Score distribution
    score_counts = {}
    for s in all_scores:
        score_counts[s] = score_counts.get(s, 0) + 1
    
    report += f"### Score Distribution\n"
    for score in sorted(score_counts.keys(), key=lambda x: 10.5 if x == "X" else float(x), reverse=True):
        count = score_counts[score]
        percentage = (count / len(all_scores)) * 100
        report += f"- {score}: {count} shots ({percentage:.1f}%)\n"
    
    # Accuracy metrics
    report += (
        f"\n### Accuracy Metrics\n"
        f"- Mean distance from center: {consolidated_stats['mean_distance']:.2f} cm\n"
        f"- Maximum distance: {consolidated_stats['max_distance']:.2f} cm\n"
        f"- Standard deviation: {consolidated_stats['std_deviation']:.2f} cm\n\n"
    )
    
    # Target-by-target summary
    report += f"### Target-by-Target Summary\n"
    report += "| Target | Shots | Avg Score | Avg Distance (cm) | Best Shot | Worst Shot |\n"
    report += "|--------|-------|-----------|-------------------|-----------|------------|\n"
    
    target_stats = {}
    for filename, shot_pt, dist, score in zip(all_shot_source_images, all_shot_points, all_distances, all_scores):
        target_num = os.path.splitext(filename)[0]
        if target_num not in target_stats:
            target_stats[target_num] = {
                'shots': 0,
                'score_sum': 0,
                'distance_sum': 0,
                'best_score': 0,
                'worst_score': 11,
                'best_distance': float('inf'),
                'worst_distance': 0
            }
            
        ts = target_stats[target_num]
        ts['shots'] += 1
        
        # Convert score for calculations
        numeric_score = 10.5 if score == "X" else float(score)
        ts['score_sum'] += numeric_score
        
        ts['distance_sum'] += dist
        
        if numeric_score > ts['best_score']:
            ts['best_score'] = numeric_score
        if numeric_score < ts['worst_score']:
            ts['worst_score'] = numeric_score
            
        if dist < ts['best_distance']:
            ts['best_distance'] = dist
        if dist > ts['worst_distance']:
            ts['worst_distance'] = dist
    
    # Format target stats for the report
    for target_num, stats in sorted(target_stats.items()):
        avg_score = stats['score_sum'] / stats['shots']
        avg_distance = stats['distance_sum'] / stats['shots']
        best_score = "X" if stats['best_score'] > 10 else str(int(stats['best_score']))
        worst_score = "X" if stats['worst_score'] > 10 else str(int(stats['worst_score']))
        
        report += f"| {target_num} | {stats['shots']} | {avg_score:.2f} | {avg_distance:.2f} | {best_score} | {worst_score} |\n"
    
    # Write the consolidated report
    with open(consolidated_report_path, 'w') as f:
        f.write(report)
    print(f"Consolidated performance report saved to: {consolidated_report_path}")
    
    # Create a consolidated score distribution visualization
    plt.figure(figsize=(10, 6))
    score_labels = sorted(score_counts.keys(), key=lambda x: 10.5 if x == "X" else float(x), reverse=True)
    score_values = [score_counts[label] for label in score_labels]

    # Convert all score labels to strings for plotting
    score_labels_str = [str(label) for label in score_labels]

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(score_labels)))
    bars = plt.bar(score_labels_str, score_values, color=colors)
    
    plt.title(f"{player_name} - Score Distribution", fontsize=16)
    plt.xlabel("Score", fontsize=14)
    plt.ylabel("Number of Shots", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', fontsize=12)
    
    score_dist_path = os.path.join(output_folder, "plots", f"{player_name}_score_distribution.png")
    plt.savefig(score_dist_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    # Create a consolidated score map showing all shots with correct distance scaling
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
    
    # Choose a standard scaling factor for the consolidated map (cm to pixels)
    standard_px_per_cm = 100  # Example: 100 pixels per cm for the consolidated map
    main_center = (600, 600)  # Center point for the consolidated map
    
    # Define standard scoring rings and labels
    ring_radii_cm = [0.5, 0.7, 1.25, 2.1, 2.5, 3.5, 4.0, 4.5, 5.5, 6.5]
    ring_labels = ["X", "10", "9", "8", "7", "6", "5", "4", "3", "2"]
    
    # Draw scoring rings centered at the standard center point
    for r, label in zip(ring_radii_cm, ring_labels):
        circle = plt.Circle(main_center, r * standard_px_per_cm, color='white', 
                          fill=False, linewidth=1.5, alpha=0.7)
        ax.add_patch(circle)
        
        # Add ring labels at different angles for better readability
        for angle in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
            label_x = main_center[0] + (r * standard_px_per_cm * np.cos(angle))
            label_y = main_center[1] + (r * standard_px_per_cm * np.sin(angle))
            if angle == np.pi/4 or angle == 5*np.pi/4:
                ax.text(label_x, label_y, label, color='white', fontsize=9,
                      ha='center', va='center', weight='bold')
    
    # Create a color map for scores
    cmap = plt.get_cmap("plasma")
    norm_scores = [10.5 if s == "X" else float(s) for s in all_scores]
    max_score = 10.5
    min_score = max(0, min(norm_scores)) if norm_scores else 0
    score_range = max_score - min_score
    normalized = [(s - min_score) / score_range for s in norm_scores]
    
    # Plot all shots with correct distance scaling
    for i, ((x_cm, y_cm), score_norm, score, src_img) in enumerate(
            zip(all_shot_positions_cm, normalized, all_scores, all_shot_source_images)):
        
        # Convert cm position to pixel position in consolidated map
        x_px = main_center[0] + (x_cm * standard_px_per_cm)
        y_px = main_center[1] + (y_cm * standard_px_per_cm)
        
        # Get target number for marker shape variation
        target_num = os.path.splitext(src_img)[0]
        marker_idx = hash(target_num) % 5  # Use hash to assign marker style
        marker_style = ['o', 's', '^', 'D', '*'][marker_idx]
        
        # Plot with color based on score
        ax.plot(x_px, y_px, marker=marker_style, color=cmap(score_norm), 
              markersize=12, alpha=0.8, markeredgecolor='white', markeredgewidth=0.5)
        
        # Add shot number and score with proper positioning
        ax.text(x_px, y_px+15, f"{i+1}", color='white', fontsize=8, ha='center', 
              va='center', weight='bold', 
              bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=1))
        
        ax.text(x_px+15, y_px, f"{score}", color=cmap(score_norm), fontsize=9, 
              ha='left', va='center', weight='bold',
              bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=1))
    
    # Calculate and display MPI (Mean Point of Impact) in cm and convert to pixels
    if all_shot_positions_cm:
        mpi_x_cm = sum(p[0] for p in all_shot_positions_cm) / len(all_shot_positions_cm)
        mpi_y_cm = sum(p[1] for p in all_shot_positions_cm) / len(all_shot_positions_cm)
        
        # Convert MPI from cm to pixels
        mpi_x_px = main_center[0] + (mpi_x_cm * standard_px_per_cm)
        mpi_y_px = main_center[1] + (mpi_y_cm * standard_px_per_cm)
        
        # Draw MPI marker
        ax.plot(mpi_x_px, mpi_y_px, 'gx', markersize=15, markeredgewidth=3, label="MPI")
        
        # Calculate MPI distance from center in cm
        mpi_dist_from_center = math.sqrt(mpi_x_cm**2 + mpi_y_cm**2)
        
        # Add circle showing group size based on standard deviation
        if consolidated_stats['std_deviation'] > 0:
            group_radius_px = consolidated_stats['std_deviation'] * standard_px_per_cm
            group_circle = plt.Circle((mpi_x_px, mpi_y_px), group_radius_px, color='lime', 
                                    fill=False, linestyle='-', linewidth=2, alpha=0.7)
            ax.add_patch(group_circle)
            
            ax.text(mpi_x_px, mpi_y_px + group_radius_px + 20, 
                  f"Group: {consolidated_stats['std_deviation']*2:.2f} cm", 
                  color='lime', fontsize=10, ha='center', weight='bold',
                  bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=2))
    
    # Add a legend explaining the markers
    custom_lines = []
    custom_labels = []
    
    for target_num in sorted(set(os.path.splitext(img)[0] for img in all_shot_source_images)):
        marker_idx = hash(target_num) % 5
        marker_style = ['o', 's', '^', 'D', '*'][marker_idx]
        custom_lines.append(Line2D([0], [0], marker=marker_style, color='w', 
                                  markerfacecolor='gray', markersize=10))
        custom_labels.append(f"Target {target_num}")
    
    # Add MPI to legend
    custom_lines.append(Line2D([0], [0], marker='x', color='lime', 
                              markersize=10, markeredgewidth=2))
    custom_labels.append("Mean Point of Impact")
    
    ax.legend(custom_lines, custom_labels, loc='upper right', 
            facecolor='black', edgecolor='white', labelcolor='white')
    
    # Add title and statistics
    ax.set_title(f"{player_name} - Consolidated Score Map", color='white', fontsize=16)
    stats_text = (
        f"Total shots: {consolidated_stats['total_shots']}\n"
        f"Average score: {consolidated_stats['average_score']:.2f}\n"
        f"Mean distance: {consolidated_stats['mean_distance']:.2f} cm\n"
        f"MPI offset: {mpi_dist_from_center:.2f} cm\n"
        f"Standard dev: {consolidated_stats['std_deviation']:.2f} cm"
    )
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=12,
          color='white', verticalalignment='bottom',
          bbox=dict(boxstyle='round', facecolor='black', alpha=0.8,
                   edgecolor='white', linewidth=1))
    
    # Set proper display properties
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    
    # Save consolidated score map
    consolidated_scoremap_path = os.path.join(output_folder, "plots", f"{player_name}_consolidated_scoremap.png")
    plt.savefig(consolidated_scoremap_path, bbox_inches="tight", dpi=300, facecolor='black')
    plt.close()
    print(f"Consolidated score map saved to: {consolidated_scoremap_path}")
    
    # Create a consolidated performance heatmap across all targets
    plt.figure(figsize=(12, 10))
    
    # Use a polar heatmap to show accuracy by distance
    distances = np.array(all_distances)
    score_values = np.array([10.5 if s == "X" else float(s) for s in all_scores])
    
    # Create distance bins
    dist_bins = np.linspace(0, max(distances) + 0.5, 20)
    digitized_dists = np.digitize(distances, dist_bins)
    
    # Create score bins
    score_bins = np.arange(0, 11, 1)
    digitized_scores = np.digitize(score_values, score_bins)
    
    # Create heatmap data
    heatmap_data = np.zeros((len(dist_bins), len(score_bins)))
    for d, s in zip(digitized_dists, digitized_scores):
        heatmap_data[d-1, s-1] += 1
    
    # Plot the heatmap
    sns.heatmap(heatmap_data, cmap="YlOrRd", xticklabels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "X"])
    plt.title(f"{player_name} - Distance vs Score Distribution", fontsize=16)
    plt.xlabel("Score", fontsize=14)
    plt.ylabel("Distance from center (cm)", fontsize=14)
    
    perf_heatmap_path = os.path.join(output_folder, "plots", f"{player_name}_performance_heatmap.png")
    plt.savefig(perf_heatmap_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    print("All analysis completed successfully.")
