import pygame
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
import pyrr
import math
import ctypes

from shaders import create_shader_program, CUBE_VERTEX_SHADER, CUBE_FRAGMENT_SHADER, LIGHT_VERTEX_SHADER, LIGHT_FRAGMENT_SHADER
from text_renderer import TextRenderer
from geometry import generate_sphere_vertices, cube_vertices

# Adjustable Parameters
LIGHT_POSITION_INITIAL = [0.0, 2.0, 4.0]
LIGHT_COLOR = [1.0, 1.0, 1.0]
LIGHT_MOVE_SPEED_PER_SEC = 3.0
CUBE_SHININESS = 32.0
CUBE_SCALE = 2.5
CUBE_COLORS = {
    "phong": [1.0, 0.0, 0.0], 
    "ambient": [0.0, 0.0, 1.0], 
    "diffuse": [0.0, 1.0, 0.0], 
    "specular": [1.0, 1.0, 1.0]
}

class Renderer:
    """ Main application class that handles rendering and user input. """
    def __init__(self, width, height):
        pygame.init()
        self.screen_size = (width, height)
        pygame.display.set_mode(self.screen_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Shading Models Demo - by Google's Gemini")
        glEnable(GL_DEPTH_TEST)
        
        self.cube_shader = create_shader_program(CUBE_VERTEX_SHADER, CUBE_FRAGMENT_SHADER)
        self.light_shader = create_shader_program(LIGHT_VERTEX_SHADER, LIGHT_FRAGMENT_SHADER)
        self.text_renderer = TextRenderer()

        self.cube_vao, self.light_vao = glGenVertexArrays(2)
        self.cube_vbo, self.light_vbo = glGenBuffers(2)
        
        glBindVertexArray(self.cube_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo)
        glBufferData(GL_ARRAY_BUFFER, cube_vertices.nbytes, cube_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        sphere_verts = generate_sphere_vertices(radius=1.0)
        self.sphere_vertex_count = len(sphere_verts) // 3
        glBindVertexArray(self.light_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.light_vbo)
        glBufferData(GL_ARRAY_BUFFER, sphere_verts.nbytes, sphere_verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        self.light_pos = np.array(LIGHT_POSITION_INITIAL, dtype=np.float32)
        self.camera_distance, self.camera_angle_x, self.camera_angle_y = 15.0, 10.0, 0.0
        self.mouse_dragging, self.last_mouse_pos = False, None
        self.clock = pygame.time.Clock()

    def run(self):
        """ Main application loop. """
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: 
                    self.mouse_dragging, self.last_mouse_pos = True, event.pos
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1: 
                    self.mouse_dragging = False
                elif event.type == pygame.MOUSEMOTION and self.mouse_dragging:
                    dx, dy = event.pos[0] - self.last_mouse_pos[0], event.pos[1] - self.last_mouse_pos[1]
                    self.camera_angle_y += dx * 0.25
                    self.camera_angle_x = max(-89.0, min(89.0, self.camera_angle_x - dy * 0.25))
                    self.last_mouse_pos = event.pos
                elif event.type == pygame.MOUSEWHEEL:
                    self.camera_distance = max(3.0, min(50.0, self.camera_distance - event.y))

            keys = pygame.key.get_pressed()
            move_amount = LIGHT_MOVE_SPEED_PER_SEC * dt
            if keys[pygame.K_LEFT]: self.light_pos[0] -= move_amount
            if keys[pygame.K_RIGHT]: self.light_pos[0] += move_amount
            if keys[pygame.K_UP]: self.light_pos[1] += move_amount
            if keys[pygame.K_DOWN]: self.light_pos[1] -= move_amount
            if keys[pygame.K_w]: self.light_pos[2] -= move_amount
            if keys[pygame.K_s]: self.light_pos[2] += move_amount

            glClearColor(0.1, 0.1, 0.1, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            rad_x, rad_y = math.radians(self.camera_angle_x), math.radians(self.camera_angle_y)
            cam_x = self.camera_distance * math.cos(rad_x) * math.sin(rad_y)
            cam_y = self.camera_distance * math.sin(rad_x)
            cam_z = self.camera_distance * math.cos(rad_x) * math.cos(rad_y)
            camera_pos = pyrr.Vector3([cam_x, cam_y, cam_z])
            view_matrix = pyrr.matrix44.create_look_at(camera_pos, pyrr.Vector3([0,0,0]), pyrr.Vector3([0,1,0]))
            projection_matrix = pyrr.matrix44.create_perspective_projection_matrix(
                45.0, self.screen_size[0]/self.screen_size[1], 0.1, 100.0)

            # Render cubes
            glUseProgram(self.cube_shader)
            glUniformMatrix4fv(glGetUniformLocation(self.cube_shader, "view"), 1, GL_FALSE, view_matrix)
            glUniformMatrix4fv(glGetUniformLocation(self.cube_shader, "projection"), 1, GL_FALSE, projection_matrix)
            glUniform3fv(glGetUniformLocation(self.cube_shader, "lightPos"), 1, self.light_pos)
            glUniform3fv(glGetUniformLocation(self.cube_shader, "viewPos"), 1, camera_pos)
            glUniform3fv(glGetUniformLocation(self.cube_shader, "lightColor"), 1, LIGHT_COLOR)
            glUniform1f(glGetUniformLocation(self.cube_shader, "shininess"), CUBE_SHININESS)
            
            positions = {
                "phong": [-7, 0, 0],
                "ambient": [-2, 0, 0],
                "diffuse": [2, 0, 0],
                "specular": [7, 0, 0]
            }
            shading_modes = {
                "phong": 0,
                "ambient": 1,
                "diffuse": 2,
                "specular": 3
            }
            
            glBindVertexArray(self.cube_vao)
            for name, pos in positions.items():
                scale_m = pyrr.matrix44.create_from_scale([CUBE_SCALE, CUBE_SCALE, CUBE_SCALE])
                trans_m = pyrr.matrix44.create_from_translation(pyrr.Vector3(pos))
                model_matrix = pyrr.matrix44.multiply(scale_m, trans_m)
                
                glUniformMatrix4fv(glGetUniformLocation(self.cube_shader, "model"), 1, GL_FALSE, model_matrix)
                glUniform3fv(glGetUniformLocation(self.cube_shader, "objectColor"), 1, CUBE_COLORS[name])
                glUniform1i(glGetUniformLocation(self.cube_shader, "shadingMode"), shading_modes[name])
                glDrawArrays(GL_TRIANGLES, 0, 36)

            # Render light source
            glUseProgram(self.light_shader)
            light_model_matrix = pyrr.matrix44.multiply(
                pyrr.matrix44.create_from_scale([0.2, 0.2, 0.2]),
                pyrr.matrix44.create_from_translation(self.light_pos)
            )
            glUniformMatrix4fv(glGetUniformLocation(self.light_shader, "model"), 1, GL_FALSE, light_model_matrix)
            glUniformMatrix4fv(glGetUniformLocation(self.light_shader, "view"), 1, GL_FALSE, view_matrix)
            glUniformMatrix4fv(glGetUniformLocation(self.light_shader, "projection"), 1, GL_FALSE, projection_matrix)
            
            glBindVertexArray(self.light_vao)
            glDrawArrays(GL_TRIANGLES, 0, self.sphere_vertex_count)

            # Render text labels
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                     
            labels = {
                "Phong": [-7, 0, 0],
                "Ambient": [-2, 0, 0],
                "Diffuse": [2, 0, 0],
                "Specular": [7, 0, 0]
            }
            for text, pos in labels.items():
                text_pos = [pos[0], pos[1] - CUBE_SCALE * 1, pos[2]]
                self.text_renderer.render_text(
                    text,
                    pyrr.Vector3(text_pos),
                    view_matrix,
                    projection_matrix,
                    scale=0.015,
                    color=[1.0, 1.0, 1.0]
                )

            glDisable(GL_BLEND)
            pygame.display.flip()

        self.text_renderer.cleanup()
        pygame.quit()