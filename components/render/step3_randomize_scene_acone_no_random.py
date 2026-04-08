import bpy
import os
import random
from pdb import set_trace
from datetime import datetime
import numpy as np
import sys

_rdir = os.path.dirname(os.path.abspath(__file__))
if _rdir not in sys.path:
    sys.path.insert(0, _rdir)
from asset_paths import get_bg_root, get_mat_root, get_table_root

bg_exr_root = get_bg_root()
table_gltf_root = get_table_root()
mat_gltf_root = get_mat_root()

def load_blend_scene(blender_file):
    bpy.ops.wm.open_mainfile(filepath=blender_file)

def randomize_scene(save_root):
    candidate_bg = os.listdir(bg_exr_root)
    candidate_table = os.listdir(table_gltf_root)
    candidate_mat = os.listdir(mat_gltf_root)

    rand_bg = random.choice(candidate_bg)
    rand_table = random.choice(candidate_table)
    rand_mat = random.choice(candidate_mat)

    print(f"Selected BG: {rand_bg}, Table: {rand_table}, Mat: {rand_mat}")

    # === Ensure scene has World ===
    if bpy.context.scene.world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    else:
        world = bpy.context.scene.world

    # === Ensure world uses nodes ===
    if world.use_nodes is False:
        world.use_nodes = True

    # === Background ===
    background_node = bpy.context.scene.world.node_tree.nodes["Background"]
    environment_texture_node = bpy.context.scene.world.node_tree.nodes.new("ShaderNodeTexEnvironment")
    bpy.context.scene.world.node_tree.links.new(environment_texture_node.outputs["Color"], background_node.inputs["Color"])

    rand_bg_path = os.path.join(bg_exr_root, rand_bg)
    background_hdri = bpy.data.images.load(rand_bg_path)
    environment_texture_node.image = background_hdri

    # === Reference box ===
    # datagen layout uses this reference
    # reference_box = bpy.data.objects["shape_27"].children[0]
    # SIM1-DataGen generated scenes use shape_26 as table reference
    reference_box = bpy.data.objects["shape_26"].children[0] # shape_26 is table now
    reference_box_aabb = np.array(reference_box.bound_box)
    reference_box_x_scale = reference_box_aabb[:,0].max()-reference_box_aabb[:,0].min()
    reference_box_y_scale = reference_box_aabb[:,1].max()-reference_box_aabb[:,1].min()
    reference_box_z_scale = reference_box_aabb[:,2].max()-reference_box_aabb[:,2].min()
    print(reference_box_x_scale)
    print(reference_box_y_scale)
    print(reference_box_z_scale)

    # set_trace()
    rand_table_path = os.path.join(table_gltf_root, rand_table, rand_table)
    bpy.ops.import_scene.gltf(filepath = rand_table_path)

    # with randomization
    # rand_table_name = rand_table.split(".")[0]
    # rand_table_name = "_".join(rand_table_name.split("_")[:-1])
    # print(rand_table_name)
    
    # without randomization
    rand_table_name = "wooden_table_02"

    bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    bpy.data.objects[rand_table_name].location = reference_box.location
    rand_table_aabb = np.array(bpy.data.objects[rand_table_name].bound_box)
    rand_table_x_scale = rand_table_aabb[:,0].max()-rand_table_aabb[:,0].min()
    rand_table_y_scale = rand_table_aabb[:,1].max()-rand_table_aabb[:,1].min()
    rand_table_z_scale = rand_table_aabb[:,2].max()-rand_table_aabb[:,2].min()
    print(rand_table_x_scale)
    print(rand_table_y_scale)
    print(rand_table_z_scale)

    bpy.data.objects[rand_table_name].scale[0] = reference_box_x_scale / rand_table_x_scale
    bpy.data.objects[rand_table_name].scale[1] = reference_box_y_scale / rand_table_y_scale
    bpy.data.objects[rand_table_name].scale[2] = reference_box_z_scale / rand_table_z_scale

    reference_box.hide_viewport = True
    reference_box.hide_render = True

    bpy.data.objects["particles"].hide_viewport = True
    bpy.data.objects["particles"].hide_render = True

    # try:
    # bpy.data.objects["instance_0.019"].hide_viewport = True
    # bpy.data.objects["instance_0.019"].hide_render = True
    # except:
    #     pass

    for obj in bpy.data.objects:
        if obj.name.startswith("box") or obj.name.startswith("mesh_") or obj.name.startswith("particles"):
            obj.hide_viewport = True
            obj.hide_render = True

    # set_trace()

    # === Random material ===
    rand_mat_path = os.path.join(mat_gltf_root, rand_mat, rand_mat)
    bpy.ops.import_scene.gltf(filepath=rand_mat_path)

    # Pick newly imported material
    # from pdb import set_trace
    # set_trace()
    imported_mats = [m for m in bpy.data.materials if m.name in rand_mat.split(".")[0]]
    if imported_mats:
        new_mat = imported_mats[0]
    else:
        raise ValueError(f"No material found in {rand_mat_path}")

    # === Replace cloth material ===
    cloth = bpy.data.objects["triangles"]
    cloth_mesh = cloth.data
    
    # variant with background (legacy tweak)
    # === UV unwrap ===
    cloth.select_set(True)
    bpy.ops.object.editmode_toggle()
    bpy.ops.uv.cube_project(cube_size=2)
    bpy.ops.object.editmode_toggle()
    cloth.select_set(False)

    # Remove old slots / assign new material
    cloth_mesh.materials.clear()
    cloth_mesh.materials.append(new_mat)

    # Tune material settings
    if "Principled BSDF" in new_mat.node_tree.nodes:
        bsdf_node = new_mat.node_tree.nodes["Principled BSDF"]
        bsdf_node.inputs["Sheen Weight"].default_value = 0.0

    # === Remove existing cameras (optional) ===
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

    # === Three camera definitions ===
    cam_configs = [
        {
            "name": "Camera",
            "location": (1.75, 1.5, 1.5),
            "rotation_euler": (np.radians(60), 0, np.radians(145)),
        },
        {
            "name": "Camera.001",
            "location": (-1.5, 1.75, 1.5),
            "rotation_euler": (np.radians(60), 0, np.radians(-45)),
        },
        {
            "name": "Camera.002",
            "location": (0.0, -2.0, 2.0),
            "rotation_euler": (np.radians(75), 0, np.radians(90)),
        },
        { 
            "name": "Camera.003",  # add toggle if needed
            "location": (0.0, -2.0, 2.0),
            "rotation_euler": (np.radians(75), 0, np.radians(90)),
        }
    ]

    # === Create cameras ===
    for cfg in cam_configs:
        cam_data = bpy.data.cameras.new(name=cfg["name"])
        cam_obj = bpy.data.objects.new(cfg["name"], cam_data)
        bpy.context.collection.objects.link(cam_obj)
        cam_obj.location = cfg["location"]
        cam_obj.rotation_euler = cfg["rotation_euler"]
        print(f"{cfg['name']} added at {cfg['location']}")

    # === Set active camera ===
    bpy.context.scene.camera = bpy.data.objects["Camera"]
    print("Active camera set to 'Camera'.")

    # === Black material on instance_* objects ===
    black_mat_name = "AutoBlackMaterial"
    if black_mat_name in bpy.data.materials:
        black_mat = bpy.data.materials[black_mat_name]
    else:
        black_mat = bpy.data.materials.new(name=black_mat_name)
        black_mat.use_nodes = True
        bsdf = black_mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)
        bsdf.inputs["Roughness"].default_value = 0.5

    for obj in bpy.data.objects:
        if obj.name.startswith("instance") and obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(black_mat)
            print(f"Applied black material to: {obj.name}")

    # === Save ===
    os.makedirs(save_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")

    save_path = os.path.join(save_root, f"{timestamp}.blend")
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=save_path)
    print(f" Scene saved to {save_path}")
    return save_path


# def randomize_scene(save_root):
#     candidate_bg = os.listdir(bg_exr_root)
#     candidate_table = os.listdir(table_gltf_root)
#     candidate_mat = os.listdir(mat_gltf_root)

#     rand_bg = random.choice(candidate_bg)
#     rand_table = random.choice(candidate_table)
#     rand_mat = random.choice(candidate_mat)

#     print(f"Selected BG: {rand_bg}, Table: {rand_table}, Mat: {rand_mat}")

#     # === Ensure scene has World ===
#     if bpy.context.scene.world is None:
#         world = bpy.data.worlds.new("World")
#         bpy.context.scene.world = world
#     else:
#         world = bpy.context.scene.world

#     if world.use_nodes is False:
#         world.use_nodes = True

#     # === Background ===
#     background_node = bpy.context.scene.world.node_tree.nodes["Background"]
#     environment_texture_node = bpy.context.scene.world.node_tree.nodes.new("ShaderNodeTexEnvironment")
#     bpy.context.scene.world.node_tree.links.new(environment_texture_node.outputs["Color"], background_node.inputs["Color"])

#     rand_bg_path = os.path.join(bg_exr_root, rand_bg)
#     background_hdri = bpy.data.images.load(rand_bg_path)
#     environment_texture_node.image = background_hdri

#     # === Reference box ===
#     reference_box = bpy.data.objects["shape_27"].children[0]  # datagen scene
#     # reference_box = bpy.data.objects["shape_26"].children[0]  # SIM1-DataGen scene (switch as needed)
#     reference_box_aabb = np.array(reference_box.bound_box)
#     reference_box_x_scale = reference_box_aabb[:,0].max() - reference_box_aabb[:,0].min()
#     reference_box_y_scale = reference_box_aabb[:,1].max() - reference_box_aabb[:,1].min()
#     reference_box_z_scale = reference_box_aabb[:,2].max() - reference_box_aabb[:,2].min()
#     print(f"Reference box scales: x={reference_box_x_scale:.3f}, y={reference_box_y_scale:.3f}, z={reference_box_z_scale:.3f}")

#     # === Import table (robust diff) ===
#     # Snapshot objects before import
#     existing_objs = set(bpy.data.objects.keys())
    
#     rand_table_path = os.path.join(table_gltf_root, rand_table, rand_table)
#     bpy.ops.import_scene.gltf(filepath=rand_table_path)
    
#     # New objects excluding empties/cameras
#     new_objs = [obj for obj in bpy.data.objects if obj.name not in existing_objs and obj.type == 'MESH']
    
#     if not new_objs:
#         # Fallback: include empties
#         new_objs = [obj for obj in bpy.data.objects if obj.name not in existing_objs]
#         if not new_objs:
#             raise RuntimeError(f"No new objects imported from {rand_table_path}")
    
#     # First mesh as table root
#     table_obj = new_objs[0]
#     print(f"Imported table object name: '{table_obj.name}' (from {rand_table})")

#     # Apply transforms
#     bpy.ops.object.select_all(action='DESELECT')
#     table_obj.select_set(True)
#     bpy.context.view_layer.objects.active = table_obj
#     bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
#     bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

#     # Align to reference box
#     table_obj.location = reference_box.location
#     table_aabb = np.array(table_obj.bound_box)
#     table_x_scale = table_aabb[:,0].max() - table_aabb[:,0].min()
#     table_y_scale = table_aabb[:,1].max() - table_aabb[:,1].min()
#     table_z_scale = table_aabb[:,2].max() - table_aabb[:,2].min()
#     print(f"Table scales: x={table_x_scale:.3f}, y={table_y_scale:.3f}, z={table_z_scale:.3f}")

#     # Scale to fit
#     table_obj.scale[0] = reference_box_x_scale / table_x_scale if table_x_scale > 1e-5 else 1.0
#     table_obj.scale[1] = reference_box_y_scale / table_y_scale if table_y_scale > 1e-5 else 1.0
#     table_obj.scale[2] = reference_box_z_scale / table_z_scale if table_z_scale > 1e-5 else 1.0

#     # Hide reference box
#     reference_box.hide_viewport = True
#     reference_box.hide_render = True

#     # Hide particles / helpers
#     for obj_name in ["particles", "box", "mesh_"]:
#         if obj_name in bpy.data.objects:
#             obj = bpy.data.objects[obj_name]
#             obj.hide_viewport = True
#             obj.hide_render = True

#     # Hide all instance_*
#     for obj in bpy.data.objects:
#         if obj.name.startswith("instance") and obj.type == 'MESH':
#             obj.hide_viewport = True
#             obj.hide_render = True

#     # === Random material ===
#     # (material import unchanged; consider set-diff for robustness)
#     rand_mat_path = os.path.join(mat_gltf_root, rand_mat, rand_mat)
#     bpy.ops.import_scene.gltf(filepath=rand_mat_path)
    
#     # Pick from newly imported materials
#     new_mats = [m for m in bpy.data.materials if m.name.startswith(rand_mat.split(".")[0][:10])]  # prefix match
#     if not new_mats:
#         raise ValueError(f"No material found after importing {rand_mat_path}")
#     new_mat = new_mats[0]

#     # Replace cloth material
#     cloth = bpy.data.objects["triangles"]
#     cloth.data.materials.clear()
#     cloth.data.materials.append(new_mat)

#     if "Principled BSDF" in new_mat.node_tree.nodes:
#         bsdf_node = new_mat.node_tree.nodes["Principled BSDF"]
#         bsdf_node.inputs["Sheen Weight"].default_value = 0.0

#     # === Cameras (unchanged) ===
#     # ... camera creation unchanged ...

#     # === Save ===
#     os.makedirs(save_root, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
#     save_path = os.path.join(save_root, f"{timestamp}.blend")
#     bpy.ops.file.pack_all()
#     bpy.ops.wm.save_as_mainfile(filepath=save_path)
#     print(f"Scene saved to {save_path}")
#     return save_path

def step_03(blender_file, save_root, record_id):
    load_blend_scene(blender_file)
    save_path = os.path.join(save_root, 'blend_out', record_id)
    return randomize_scene(save_path)