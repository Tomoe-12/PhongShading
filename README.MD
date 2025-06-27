# Phong Shading Models Demo

This project is an interactive OpenGL visualization of different shading models (Phong, Ambient, Diffuse, Specular) using Python, PyOpenGL, and pygame. It displays four cubes, each demonstrating a different shading component, with a movable light source and camera controls. Text labels are rendered in 3D space as billboards.

## Features

- Phong, Ambient, Diffuse, and Specular shading: Each cube shows a different lighting model.
- Interactive camera: Orbit and zoom using mouse drag and scroll.
- Movable light source: Move the light in 3D using arrow keys and W/S.
- 3D text billboards: Labels under each cube always face the camera.
- Clean modular code: Geometry, shaders, rendering, and text are separated for clarity.

## Controls

- Rotate camera: Left mouse button drag
- Zoom camera: Mouse wheel
- Move light: Arrow keys (X/Y), W/S (Z)
- Quit: Close window

## Requirements

- Python 3.8+
- pygame >= 2.0.0
- PyOpenGL >= 3.1.0
- numpy >= 1.19.0
- pyrr >= 0.10.3

Install dependencies with:

    pip install -r requirements.txt

## Running

To start the demo, run:

    python main.py

or

    python gemini6.py

Both files launch the same visualization.

## File Structure

- main.py: Entry point for the application.
- renderer.py: Main rendering loop and user input.
- geometry.py: Cube and sphere geometry generation.
- shaders.py: GLSL shader source and compilation helpers.
- text_renderer.py: 3D text rendering as billboards.
- gemini6.py: Alternative all-in-one script (for reference).
- requirements.txt: Python dependencies.

## Credits

- Developed using pygame, PyOpenGL, numpy, and pyrr.
- Shading models and visualization inspired by classic computer graphics techniques.

Enjoy exploring real-time shading!