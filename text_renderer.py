import pygame
from OpenGL.GL import *
import numpy as np
import pyrr
import ctypes

from shaders import create_shader_program, TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER

class TextRenderer:
    """ Handles all text rendering operations. """
    def __init__(self, font_size=48):
        pygame.font.init()
        self.font = pygame.font.Font(None, font_size)
        self.shader = create_shader_program(TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER)
        self.text_cache = {}

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # Quad vertices: PosX, PosY, PosZ, TexU, TexV
        quad_vertices = np.array([
            -0.5,-0.5,0,0,0, 
            0.5,-0.5,0,1,0, 
            0.5,0.5,0,1,1, 
            -0.5,-0.5,0,0,0, 
            0.5,0.5,0,1,1, 
            -0.5,0.5,0,0,1
        ], dtype=np.float32)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    def create_text_texture(self, text):
        """ Creates or retrieves a cached texture for a given string. """
        if text in self.text_cache:
            return self.text_cache[text]
        
        font_surface = self.font.render(text, True, (255, 255, 255))
        width, height = font_surface.get_size()
        rgba_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        rgba_surface.blit(font_surface, (0, 0))
        texture_data = pygame.image.tostring(rgba_surface, "RGBA", True)
        
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        self.text_cache[text] = (texture_id, width, height)
        return texture_id, width, height

    def render_text(self, text, position, view_matrix, projection_matrix, scale=0.01, color=(1.0, 1.0, 1.0)):
        """ Renders text as a camera-facing billboard at a given world position. """
        texture_id, width, height = self.create_text_texture(text)
        model_matrix = pyrr.matrix44.create_identity()

        # Create billboard effect
        billboard_rot_matrix = view_matrix.copy()
        billboard_rot_matrix[3][0] = 0
        billboard_rot_matrix[3][1] = 0
        billboard_rot_matrix[3][2] = 0
        billboard_rot_matrix = pyrr.matrix44.inverse(billboard_rot_matrix)

        # Build model matrix
        scale_matrix = pyrr.matrix44.create_from_scale([width * scale, height * scale, 1.0])
        model_matrix = pyrr.matrix44.multiply(scale_matrix, model_matrix)
        model_matrix = pyrr.matrix44.multiply(billboard_rot_matrix, model_matrix)
        trans_matrix = pyrr.matrix44.create_from_translation(position)
        model_matrix = pyrr.matrix44.multiply(model_matrix, trans_matrix)

        glUseProgram(self.shader)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, model_matrix)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, view_matrix)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, projection_matrix)
        glUniform3fv(glGetUniformLocation(self.shader, "textColor"), 1, color)
        glUniform1i(glGetUniformLocation(self.shader, "textTexture"), 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6) 
        
    def cleanup(self):
        """ Deletes all generated GL objects and quits pygame.font. """
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteTextures([t[0] for t in self.text_cache.values()])
        glDeleteProgram(self.shader)
        pygame.font.quit()