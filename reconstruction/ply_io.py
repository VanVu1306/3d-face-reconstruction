# reconstruction/ply_io.py
import numpy as np

def write_textured_ply(filename, vertices, triangles, vertex_colors):
    vertex_colors = np.clip(vertex_colors, 0, 255).astype(np.uint8)

    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(triangles)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for i in range(len(vertices)):
            v = vertices[i]
            c = vertex_colors[i]
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

        for tri in triangles:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

    print(f" File saved: {filename}")
    print(f"  Vertices: {len(vertices)}, Faces: {len(triangles)}")

    color_stats = f"Color range: [{np.min(vertex_colors)}-{np.max(vertex_colors)}]"
    color_stats += f", Mean: {np.mean(vertex_colors):.1f}"
    print(f"  {color_stats}")

    return True
