import os
import random
import glob
from typing import Tuple, List
from PIL import Image
from problem import ClusteringProblem
import folium
import numpy as np


def generate_colors(n: int) -> List[Tuple[int, int, int]]:
    colors = [(random.randint(30, 255), random.randint(30, 255), random.randint(30, 255)) for i in range(n)]
    return colors


def generate_marker(color: Tuple[int, int, int], output_path: str, asset_path: str = "./assets/marker.png"):
    # Load the image
    img = Image.open(asset_path)

    # Define the colors
    dark_color = color  # Replace with your desired (r, g, b) values
    light_color = (color[0] - 30, color[1] - 30, color[2] - 30)  # Replace with your desired (r+10, g+10, b+10) values

    # Iterate over each pixel and change the color based on brightness
    for x in range(img.width):
        for y in range(img.height):
            r, g, b, a = img.getpixel((x, y))  # Get the pixel color and alpha channel
            if a == 0:
                continue  # Skip transparent pixels
            brightness = (r + g + b) // 3  # Calculate brightness
            if brightness < 239:  # Check brightness
                img.putpixel((x, y), light_color + (a,))  # Set pixel to light color with original alpha
            else:
                img.putpixel((x, y), dark_color + (a,))  # Set pixel to dark color with original alpha

    # Save the modified image
    img.save(output_path)


class Renderer:
    def __init__(self, markers_dir: str = "./markers"):
        if not os.path.exists(markers_dir):
            os.mkdir(markers_dir)
        self.markers: List[str] = glob.glob(f"{markers_dir}/*.png")
        self.markers_count: int = len(self.markers)
        self.markers_dir: str = markers_dir

    def generate_needed_markers(self, clusters_count: int):
        if clusters_count > self.markers_count:
            new_colors = generate_colors(clusters_count - self.markers_count)
            for color in new_colors:
                generate_marker(color, f"{self.markers_dir}/{self.markers_count}.png")
                self.markers.append(f"{self.markers_dir}/{self.markers_count}.png")
                self.markers_count += 1

    def render(self, problem: ClusteringProblem, output_path: str):
        self.generate_needed_markers(problem.clusters_count)
        rendered_map = folium.Map(location=np.mean(problem.positions, axis=0),
                                  zoom_start=4)

        for i in range(problem.positions_count):
            icon = folium.CustomIcon(icon_image=self.markers[problem.assignments[i]], icon_size=(25, 41))
            folium.Marker(location=problem.positions[i], icon=icon,
                          popup=f'Cluster ID: {problem.assignments[i]}').add_to(rendered_map)

        rendered_map.save(output_path)
