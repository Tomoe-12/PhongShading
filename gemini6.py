import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader, ShaderCompilationError
import numpy as np
import pyrr
import math
import ctypes

# --- Helper function for robust shader compilation ---
def create_shader_program(vertex_shader_code, fragment_shader_code):
    """ Compiles GLSL shaders and links them into a program. """
    try:
        vertex_shader = compileShader(vertex_shader_code, GL_VERTEX_SHADER)
        fragment_shader = compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)
        program = compileProgram(vertex_shader, fragment_shader)
        return program
    except ShaderCompilationError as e:
        print("\n--- SHADER COMPILATION ERROR ---")
        print(e)
        # Print the shader code with line numbers for easier debugging
        print("\n--- Vertex Shader ---")
        for i, line in enumerate(vertex_shader_code.splitlines()):
            print(f"{i+1}: {line}")
        print("\n--- Fragment Shader ---")
        for i, line in enumerate(fragment_shader_code.splitlines()):
            print(f"{i+1}: {line}")
        raise

# --- Adjustable Parameters ---
LIGHT_POSITION_INITIAL = [0.0, 2.0, 4.0]
LIGHT_COLOR = [1.0, 1.0, 1.0]
LIGHT_MOVE_SPEED_PER_SEC = 3.0
CUBE_SHININESS = 32.0
CUBE_SCALE = 2.5
CUBE_COLORS = {"phong": [1.0, 0.0, 0.0], "ambient": [0.0, 0.0, 1.0], "diffuse": [0.0, 1.0, 0.0], "specular": [1.0, 1.0, 1.0]}

# --- Cube Vertex Shader (Unchanged) ---
CUBE_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
out vec3 FragPos;
out vec3 Normal;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

# --- Cube Fragment Shader (Unchanged) ---
CUBE_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
in vec3 FragPos;
in vec3 Normal;
uniform vec3 objectColor;
uniform float shininess;
uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform int shadingMode;
void main()
{
    float ambientStrength = 0.1;
    if (shadingMode == 1) { // Ambient only mode
        ambientStrength = 0.8;
    }
    vec3 ambient = ambientStrength * lightColor;

    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = spec * lightColor;
    
    vec3 result;
    if (shadingMode == 0) { // Full Phong
        result = (ambient + diffuse + specular) * objectColor;
    } else if (shadingMode == 1) { // Ambient Only
        result = ambient * objectColor;
    } else if (shadingMode == 2) { // Diffuse Only
        result = diffuse * objectColor;
    } else if (shadingMode == 3) { // Specular Only
        vec3 baseColor = objectColor * 0.05;
        result = baseColor + (specular * vec3(1.0));
    } else { // Default to Full Phong
        result = (ambient + diffuse + specular) * objectColor;
    }
    
    FragColor = vec4(result, 1.0);
}
"""

# --- Light Shaders (Unchanged) ---
LIGHT_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() { gl_Position = projection * view * model * vec4(aPos, 1.0); }
"""

LIGHT_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
void main() { FragColor = vec4(1.0); }
"""

# --- Shaders for rendering text billboards ---
TEXT_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;
out vec2 TexCoords;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    TexCoords = aTexCoords;
}
"""

TEXT_FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoords;
out vec4 FragColor;
uniform sampler2D textTexture;
uniform vec3 textColor;
void main()
{
    vec4 sampled = texture(textTexture, TexCoords);
    FragColor = vec4(textColor, sampled.a);
}
"""

# --- Cube vertex data (Position + Normal) ---
cube_vertices = np.array([-0.5,-0.5,-0.5,0,0,-1,.5,-0.5,-0.5,0,0,-1,.5,.5,-0.5,0,0,-1,.5,.5,-0.5,0,0,-1,-.5,.5,-0.5,0,0,-1,-.5,-0.5,-0.5,0,0,-1,-.5,-0.5,.5,0,0,1,.5,-0.5,.5,0,0,1,.5,.5,.5,0,0,1,.5,.5,.5,0,0,1,-.5,.5,.5,0,0,1,-.5,-0.5,.5,0,0,1,-.5,.5,.5,-1,0,0,-.5,.5,-.5,-1,0,0,-.5,-.5,-.5,-1,0,0,-.5,-.5,-.5,-1,0,0,-.5,-.5,.5,-1,0,0,-.5,.5,.5,-1,0,0,.5,.5,.5,1,0,0,.5,.5,-.5,1,0,0,.5,-.5,-.5,1,0,0,.5,-.5,-.5,1,0,0,.5,-.5,.5,1,0,0,.5,.5,.5,1,0,0,-.5,-0.5,-0.5,0,-1,0,.5,-0.5,-0.5,0,-1,0,.5,-0.5,.5,0,-1,0,.5,-0.5,.5,0,-1,0,-.5,-0.5,.5,0,-1,0,-.5,-0.5,-.5,0,-1,0,-.5,.5,-.5,0,1,0,.5,.5,-.5,0,1,0,.5,.5,.5,0,1,0,.5,.5,.5,0,1,0,-.5,.5,.5,0,1,0,-.5,.5,-.5,0,1,0],dtype=np.float32)

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

# --- Text Rendering Class ---
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
        quad_vertices = np.array([-0.5,-0.5,0,0,0, 0.5,-0.5,0,1,0, 0.5,0.5,0,1,1, -0.5,-0.5,0,0,0, 0.5,0.5,0,1,1, -0.5,0.5,0,0,1], dtype=np.float32)
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

            # Start with an identity matrix for the model
            model_matrix = pyrr.matrix44.create_identity()

            # --- The key correction is here ---
            # We want the billboard to face the camera, but not be affected by its position.
            # So, we take the view matrix, remove its translation component, and then
            # take its inverse to get a matrix that only contains the camera's rotation.
            billboard_rot_matrix = view_matrix.copy()
            billboard_rot_matrix[3][0] = 0
            billboard_rot_matrix[3][1] = 0
            billboard_rot_matrix[3][2] = 0
            billboard_rot_matrix = pyrr.matrix44.inverse(billboard_rot_matrix)

            # Now, we build the model matrix in the correct order:
            # 1. Scale the quad to the desired size.
            # 2. Rotate it to face the camera.
            # 3. Translate it to the desired world position.
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
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: self.mouse_dragging, self.last_mouse_pos = True, event.pos
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1: self.mouse_dragging = False
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

            glClearColor(0.1, 0.1, 0.1, 1.0); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            rad_x,rad_y=math.radians(self.camera_angle_x),math.radians(self.camera_angle_y)
            cam_x=self.camera_distance*math.cos(rad_x)*math.sin(rad_y)
            cam_y=self.camera_distance*math.sin(rad_x)
            cam_z=self.camera_distance*math.cos(rad_x)*math.cos(rad_y)
            camera_pos=pyrr.Vector3([cam_x,cam_y,cam_z])
            view_matrix=pyrr.matrix44.create_look_at(camera_pos,pyrr.Vector3([0,0,0]),pyrr.Vector3([0,1,0]))
            projection_matrix=pyrr.matrix44.create_perspective_projection_matrix(45.0,self.screen_size[0]/self.screen_size[1],0.1,100.0)

            glUseProgram(self.cube_shader)
            glUniformMatrix4fv(glGetUniformLocation(self.cube_shader,"view"),1,GL_FALSE,view_matrix)
            glUniformMatrix4fv(glGetUniformLocation(self.cube_shader,"projection"),1,GL_FALSE,projection_matrix)
            glUniform3fv(glGetUniformLocation(self.cube_shader,"lightPos"),1,self.light_pos)
            glUniform3fv(glGetUniformLocation(self.cube_shader,"viewPos"),1,camera_pos)
            glUniform3fv(glGetUniformLocation(self.cube_shader,"lightColor"),1,LIGHT_COLOR)
            glUniform1f(glGetUniformLocation(self.cube_shader,"shininess"),CUBE_SHININESS)
            
            positions={"phong":[-7,0,0],"ambient":[-2,0,0],"diffuse":[2,0,0],"specular":[7,0,0]}
            shading_modes={"phong":0,"ambient":1,"diffuse":2,"specular":3}
            
            glBindVertexArray(self.cube_vao)
            for name,pos in positions.items():
                scale_m = pyrr.matrix44.create_from_scale([CUBE_SCALE, CUBE_SCALE, CUBE_SCALE])
                trans_m = pyrr.matrix44.create_from_translation(pyrr.Vector3(pos))
                model_matrix = pyrr.matrix44.multiply(scale_m, trans_m)
                
                glUniformMatrix4fv(glGetUniformLocation(self.cube_shader,"model"),1,GL_FALSE,model_matrix)
                glUniform3fv(glGetUniformLocation(self.cube_shader,"objectColor"),1,CUBE_COLORS[name])
                glUniform1i(glGetUniformLocation(self.cube_shader,"shadingMode"),shading_modes[name])
                glDrawArrays(GL_TRIANGLES,0,36)

            glUseProgram(self.light_shader)
            light_model_matrix=pyrr.matrix44.multiply(pyrr.matrix44.create_from_scale([0.2,0.2,0.2]),pyrr.matrix44.create_from_translation(self.light_pos))
            glUniformMatrix4fv(glGetUniformLocation(self.light_shader,"model"),1,GL_FALSE,light_model_matrix)
            glUniformMatrix4fv(glGetUniformLocation(self.light_shader,"view"),1,GL_FALSE,view_matrix)
            glUniformMatrix4fv(glGetUniformLocation(self.light_shader,"projection"),1,GL_FALSE,projection_matrix)
            
            glBindVertexArray(self.light_vao)
            glDrawArrays(GL_TRIANGLES,0,self.sphere_vertex_count)

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                     
            labels = {"Phong":[-7,0,0], "Ambient":[-2,0,0], "Diffuse":[2,0,0], "Specular":[7,0,0]}
            for text, pos in labels.items():
                # --- THIS IS THE MODIFIED LINE ---
                # Position text below the cube for clarity
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

if __name__ == "__main__":
    try:
        Renderer(1280, 720).run()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pygame.get_init():
            pygame.quit()
