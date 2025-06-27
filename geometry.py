import numpy as np
import math

# Cube vertex data (Position + Normal)
cube_vertices = np.array([
    -0.5,-0.5,-0.5,0,0,-1,.5,-0.5,-0.5,0,0,-1,.5,.5,-0.5,0,0,-1,.5,.5,-0.5,0,0,-1,-.5,.5,-0.5,0,0,-1,-.5,-0.5,-0.5,0,0,-1,
    -.5,-0.5,.5,0,0,1,.5,-0.5,.5,0,0,1,.5,.5,.5,0,0,1,.5,.5,.5,0,0,1,-.5,.5,.5,0,0,1,-.5,-0.5,.5,0,0,1,
    -.5,.5,.5,-1,0,0,-.5,.5,-.5,-1,0,0,-.5,-.5,-.5,-1,0,0,-.5,-.5,-.5,-1,0,0,-.5,-.5,.5,-1,0,0,-.5,.5,.5,-1,0,0,
    .5,.5,.5,1,0,0,.5,.5,-.5,1,0,0,.5,-.5,-.5,1,0,0,.5,-.5,-.5,1,0,0,.5,-.5,.5,1,0,0,.5,.5,.5,1,0,0,
    -.5,-0.5,-0.5,0,-1,0,.5,-0.5,-0.5,0,-1,0,.5,-0.5,.5,0,-1,0,.5,-0.5,.5,0,-1,0,-.5,-0.5,.5,0,-1,0,-.5,-0.5,-.5,0,-1,0,
    -.5,.5,-.5,0,1,0,.5,.5,-.5,0,1,0,.5,.5,.5,0,1,0,.5,.5,.5,0,1,0,-.5,.5,.5,0,1,0,-.5,.5,-.5,0,1,0
], dtype=np.float32)

def generate_sphere_vertices(radius=1.0, sectors=24, stacks=18):
    """ Generates vertices for a sphere. Used for the light source object. """
    vertices = []
    indices = []
    for i in range(stacks + 1):
        phi = math.pi / 2.0 - i * math.pi / float(stacks)
        y = radius * math.sin(phi)
        r = radius * math.cos(phi)
        for j in range(sectors + 1):
            theta = j * 2.0 * math.pi / float(sectors)
            x = r * math.cos(theta)
            z = r * math.sin(theta)
            vertices.extend([x, y, z])
    
    for i in range(stacks):
        for j in range(sectors):
            indices.extend([i * (sectors + 1) + j, (i + 1) * (sectors + 1) + j, i * (sectors + 1) + j + 1])
            indices.extend([(i + 1) * (sectors + 1) + j, (i + 1) * (sectors + 1) + j + 1, i * (sectors + 1) + j + 1])

    final_vertices = np.array(vertices, dtype=np.float32)[indices]
    return final_vertices.flatten()