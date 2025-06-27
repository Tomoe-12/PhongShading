from OpenGL.GL.shaders import compileProgram, compileShader, ShaderCompilationError
from OpenGL.GL import GL_VERTEX_SHADER, GL_FRAGMENT_SHADER

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

# --- Cube Vertex Shader ---
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

# --- Cube Fragment Shader ---
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

# --- Light Shaders ---
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

# --- Text Shaders ---
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