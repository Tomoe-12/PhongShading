import pygame
from renderer import Renderer

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