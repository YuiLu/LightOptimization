"""Convert the official Blender Classroom geometry to the URP scale fixture.

Run Blender in background with the original classroom.blend (all linked assets
beside it). No Blender image or illumination is used by the experiment.
"""
from pathlib import Path
import array
import json
import shutil
import struct
import sys

import bpy


def unity_xyz(v):
    return (v.x, v.z, v.y)


def convert(output):
    output.mkdir(parents=True, exist_ok=True)
    if sys.byteorder != "little":
        raise RuntimeError("Fixture mesh format requires little-endian floats/indices")
    scene = bpy.data.scenes["_mainScene"]
    bpy.context.window.scene = scene
    bpy.context.window.view_layer = scene.view_layers["interior"]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    buckets = {}
    lights = []
    excluded = []

    for instance in depsgraph.object_instances:
        obj = instance.object
        if obj.hide_render:
            continue
        transform = instance.matrix_world.copy()
        if obj.type == "LIGHT":
            lamp = dict(name=obj.name, watts=obj.data.energy, position=unity_xyz(transform.translation), type=obj.data.type)
            (lights if obj.data.type == "POINT" else excluded).append(lamp)
            continue
        if obj.type != "MESH":
            continue
        mesh = obj.data
        mesh.calc_loop_triangles()
        required_normals = mesh.corner_normals
        normal_transform = transform.to_3x3().inverted_safe().transposed()
        mirrored = transform.to_3x3().determinant() < 0
        uv = mesh.uv_layers.active
        local_indices = {}
        for tri in mesh.loop_triangles:
            mat = obj.material_slots[tri.material_index].material if tri.material_index < len(obj.material_slots) else None
            name = mat.name if mat else "neutral"
            # Portals belonged to Cycles' daylight setup, and transmission is not part of this port.
            if "glass" in name.lower() or "portal" in name.lower():
                continue
            if name not in buckets:
                buckets[name] = (mat, array.array("f"), array.array("I"))
            _, vertices, indices = buckets[name]
            loops = tri.loops if mirrored else reversed(tri.loops)
            for li in loops:
                vertex = mesh.loops[li].vertex_index
                normal = required_normals[li].vector
                wp = transform @ mesh.vertices[vertex].co
                tex = tuple(uv.data[li].uv) if uv else (wp.x * .45 + wp.z * .33, wp.y * .45 + wp.z * .29)
                key = (name, vertex, *normal, *tex)
                index = local_indices.get(key)
                if index is None:
                    index = local_indices[key] = len(vertices) // 8
                    vertices.extend((*unity_xyz(wp), *unity_xyz((normal_transform @ normal).normalized()), *tex))
                indices.append(index)

    materials = []
    textures = {}
    tri_count = 0
    with (output / "room.bytes").open("wb") as handle:
        handle.write(struct.pack("<II", 0x53434C45, len(buckets)))
        for name in sorted(buckets):
            mat, vertices, indices = buckets[name]
            mat_id = len(materials)
            handle.write(struct.pack("<II", len(vertices) // 8, len(indices)))
            vertices.tofile(handle)
            indices.tofile(handle)
            tri_count += len(indices) // 3
            color = list(mat.diffuse_color) if mat else [.42, .42, .42, 1]
            nodes = list(mat.node_tree.nodes) if mat and mat.node_tree else []
            images = [n.image for n in nodes if n.type == "TEX_IMAGE" and n.image]
            images = [im for im in images if not any(k in im.name.lower() for k in ("mask", "bump", "normal", "wireframe", "displac"))]
            # Original texture pixels and UVs are retained; the Cycles procedural network isn't evaluated.
            if name == "varnishedWoodDoor":
                images.sort(key=lambda im: "brownwood" not in im.name.lower())
            tex_name = ""
            if images:
                image = images[0]
                key = (image.name, image.filepath)
                if key not in textures:
                    source = Path(bpy.path.abspath(image.filepath, library=image.library))
                    ext = source.suffix.lower()
                    if ext in {".jpg", ".jpeg", ".png"} and (image.packed_file or source.is_file()):
                        dst = output / f"tex_{len(textures):03d}{ext}"
                        if image.packed_file:
                            dst.write_bytes(image.packed_file.data)
                        else:
                            shutil.copyfile(source, dst)
                        textures[key] = dst.name
                    else:
                        textures[key] = ""
                tex_name = textures[key]
                if tex_name:
                    color = [.77, .77, .77, 1]
            diffuse = [n for n in nodes if n.type == "BSDF_DIFFUSE"]
            gloss = [n for n in nodes if n.type in {"BSDF_GLOSSY", "BSDF_ANISOTROPIC"}]
            roughness = float(gloss[0].inputs["Roughness"].default_value) if gloss else .84
            roughness = max(.19, min(.90, roughness))
            metal = .6 if any(word in name.lower() for word in ["metal", "brass", "steel"]) else .0
            if not tex_name and diffuse and not diffuse[0].inputs["Color"].is_linked:
                color = list(diffuse[0].inputs["Color"].default_value)
            emissive = .06 if name in {"ceillingLamp_light", "blackBoardLight"} else 0
            materials.append(dict(id=mat_id, name=name, color=color, smoothness=1 - roughness, metallic=metal,
                                  texture=tex_name, emissive=emissive))

    lamps = sorted(lights, key=lambda x: (x["name"], x["position"]))
    if not 1 <= len(lamps) <= 10:
        raise RuntimeError(f"Expected the complete small point-light rig; got {len(lamps)} instances")
    info = dict(version=1, source="Classroom – Christophe Seux / Blender demo files", blender=bpy.app.version_string,
                sourceUrl="https://download.blender.org/demo/test/classroom.zip", lamps=lamps, excluded=excluded,
                materials=materials, triangles=tri_count)
    (output / "room.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(lamps)} authored point lights, {tri_count} triangles, {len(materials)} materials to {output}")


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) != 1:
        raise SystemExit("Usage: blender -b classroom.blend --python export_classroom_scale.py -- <Unity/Assets/ClassroomScale>")
    convert(Path(args[0]).resolve())
