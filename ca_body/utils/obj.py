from typing import Dict, List,TextIO, Union

import numpy as np

ObjectType = Dict[str, Union[List[np.ndarray], np.ndarray]]

def load_obj(path: Union[str, TextIO], return_vn: bool = False) -> ObjectType:
    """Load wavefront OBJ from file. See https://en.wikipedia.org/wiki/Wavefront_.obj_file for file format details
    Args:
        path: Where to load the obj file from
        return_vn: Whether we should return vertex normals

    Returns:
        Dictionary with the following entries
            v: n-by-3 float32 numpy array of vertices in x,y,z format
            vt: n-by-2 float32 numpy array of texture coordinates in uv format
            vi: n-by-3 int32 numpy array of vertex indices into `v`, each defining a face.
            vti: n-by-3 int32 numpy array of vertex texture indices into `vt`, each defining a face
            vn: (if requested) n-by-3 numpy array of normals
    """

    if isinstance(path, str):
        with open(path, "r") as f:
            lines: List[str] = f.readlines()
    else:
        lines: List[str] = path.readlines()

    v = []
    vt = []
    vindices = []
    vtindices = []
    vn = []

    for line in lines:
        if line == "":
            break

        if line[:2] == "v ":
            v.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "vt":
            vt.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "vn":
            vn.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "f ":
            vindices.append([int(entry.split("/")[0]) - 1 for entry in line.split()[1:]])
            if line.find("/") != -1:
                vtindices.append([int(entry.split("/")[1]) - 1 for entry in line.split()[1:]])

    if len(vt) == 0:
        assert len(vtindices) == 0, "Tried to load an OBJ with texcoord indices but no texcoords!"
        vt = [[0.5, 0.5]]
        vtindices = [[0, 0, 0]] * len(vindices)

    # If we have mixed face types (tris/quads/etc...), we can't create a
    # non-ragged array for vi / vti.
    mixed_faces = False
    for vi in vindices:
        if len(vi) != len(vindices[0]):
            mixed_faces = True
            break

    if mixed_faces:
        vi = [np.array(vi, dtype=np.int32) for vi in vindices]
        vti = [np.array(vti, dtype=np.int32) for vti in vtindices]
    else:
        vi = np.array(vindices, dtype=np.int32)
        vti = np.array(vtindices, dtype=np.int32)

    out = {
        "v": np.array(v, dtype=np.float32),
        "vn": np.array(vn, dtype=np.float32),
        "vt": np.array(vt, dtype=np.float32),
        "vi": vi,
        "vti": vti,
    }

    if return_vn:
        assert len(out["vn"]) > 0
        return out
    else:
        out.pop("vn")
        return out