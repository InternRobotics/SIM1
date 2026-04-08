import numpy as np
import OpenGL.GL
from PIL import Image


# def dump_gl_frame_image(width, height, filename="imgui_capture.png"):
#     # Read pixels from the framebuffer (bottom-to-top order)
#     data = OpenGL.GL.glReadPixels(0, 0, width, height, OpenGL.GL.GL_RGBA, OpenGL.GL.GL_UNSIGNED_BYTE)

#     # Convert to NumPy array
#     image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))

#     # Flip vertically (OpenGL’s origin is bottom-left)
#     image = np.flipud(image)

#     # Save as image using PIL
#     Image.fromarray(image).save(filename)


def dump_gl_frame_image(width, height, filename=None):
    """
    Captures the current OpenGL framebuffer and returns it as a NumPy array.
    
    Args:
        width (int): Framebuffer width.
        height (int): Framebuffer height.
        filename (str or None): If provided, save the image to this path (for debugging).
    
    Returns:
        np.ndarray: Image array of shape (height, width, 4) in RGBA format, uint8.
    """

    # Read pixels from the framebuffer (bottom-to-top)
    data =  OpenGL.GL.glReadPixels(0, 0, width, height, OpenGL.GL.GL_RGBA, OpenGL.GL.GL_UNSIGNED_BYTE)

    # Convert to NumPy array
    image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))

    # Flip vertically (OpenGL origin is bottom-left)
    image = np.flipud(image)

    # Optionally save to file (e.g., for debugging)
    if filename is not None:
        Image.fromarray(image).save(filename)

    return image  # Always return the array