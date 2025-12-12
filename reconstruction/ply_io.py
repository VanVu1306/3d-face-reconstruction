"""
PLY file I/O utilities.

Save and load 3D mesh data in PLY (Polygon File Format).
"""

import pandas as pd
import numpy as np
import sys
from typing import Optional
from pathlib import Path


def describe_element(name: str, df: pd.DataFrame) -> list:
    """
    Generate PLY element description from DataFrame.
    
    Args:
        name: Element name ('vertex' or 'face')
        df: Pandas DataFrame with element data
    
    Returns:
        List of PLY property lines
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = [f'element {name} {len(df)}']
    
    if name == 'face':
        element.append("property list uchar int vertex_indices")
    else:
        for i in range(len(df.columns)):
            # Get data type and infer format
            dtype_str = str(df.dtypes[i])
            fmt_char = dtype_str[0]
            fmt = property_formats.get(fmt_char, 'float')
            element.append(f'property {fmt} {df.columns.values[i]}')
    
    return element


def write_ply(
    filename: str,
    points: Optional[np.ndarray] = None,
    mesh: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    as_text: bool = True,
) -> bool:
    """
    Write mesh to PLY file.
    
    Args:
        filename: Output filename (.ply extension added if missing)
        points: (N, 3) vertex positions
        mesh: (M, 3) face indices
        colors: (N, 3) optional vertex colors [0, 255]
        as_text: If True, use ASCII format; else binary
    
    Returns:
        True if successful
    
    Raises:
        ValueError: If neither points nor mesh provided
        IOError: If write fails
    """
    if points is None and mesh is None:
        raise ValueError("Must provide either points or mesh")
    
    # Add .ply extension if not present
    if not filename.endswith('.ply'):
        filename += '.ply'
    
    # Create output directory if needed
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare DataFrames
    points_df = None
    mesh_df = None
    
    if points is not None:
        points_df = pd.DataFrame(points, columns=["x", "y", "z"])
        if colors is not None:
            colors_df = pd.DataFrame(colors, columns=["red", "green", "blue"])
            points_df = pd.concat([points_df, colors_df], axis=1)
    
    if mesh is not None:
        mesh_df = pd.DataFrame(mesh, columns=["v1", "v2", "v3"])
    
    # Write header
    with open(filename, 'w') as ply:
        header = ['ply']
        
        if as_text:
            header.append('format ascii 1.0')
        else:
            header.append(f'format binary_{sys.byteorder}_endian 1.0')
        
        if points_df is not None:
            header.extend(describe_element('vertex', points_df))
        
        if mesh_df is not None:
            # Add count property
            mesh_df_copy = mesh_df.copy()
            mesh_df_copy.insert(loc=0, column="n_points", value=3)
            mesh_df_copy["n_points"] = mesh_df_copy["n_points"].astype("u1")
            header.extend(describe_element('face', mesh_df_copy))
        
        header.append('end_header')
        
        for line in header:
            ply.write(f"{line}\n")
    
    # Write data
    if as_text:
        if points_df is not None:
            points_df.to_csv(
                filename,
                sep=" ",
                index=False,
                header=False,
                mode='a',
                encoding='ascii'
            )
        
        if mesh_df is not None:
            mesh_df_copy = mesh_df.copy()
            mesh_df_copy.insert(loc=0, column="n_points", value=3)
            mesh_df_copy.to_csv(
                filename,
                sep=" ",
                index=False,
                header=False,
                mode='a',
                encoding='ascii'
            )
    else:
        # Binary format
        with open(filename, 'ab') as ply:
            if points_df is not None:
                points_df.to_records(index=False).tofile(ply)
            
            if mesh_df is not None:
                mesh_df_copy = mesh_df.copy()
                mesh_df_copy.insert(loc=0, column="n_points", value=3)
                mesh_df_copy.to_records(index=False).tofile(ply)
    
    return True
