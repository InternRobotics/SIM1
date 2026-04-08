# usd_to_blend_anim.py
# Usage:
# blender --background --python usd_to_blend_anim.py -- /path/to/input.usd /path/to/output.blend

import bpy
import sys
import os
import traceback
from typing import Dict, Any, List

def enable_usd_addon() -> None:
    """
    Enable the USD add-on (tuned for Blender 4.5)
    """
    try:
        # Check if already enabled
        if "io_scene_usd" in bpy.context.preferences.addons:
            print("✓ USD addon is already enabled.")
            return True
        
        # Try to enable
        result = bpy.ops.preferences.addon_enable(module="io_scene_usd")
        if 'FINISHED' in result:
            print("✓ USD addon enabled successfully.")
            return True
        else:
            print("⚠ Could not enable USD addon via operator.")
            return False
            
    except Exception as e:
        print("⚠ Could not enable USD addon automatically:", e)
        print("Please ensure the addon is installed (Edit > Preferences > Add-ons > 'USD')")
        return False

def detect_usd_import_parameters() -> Dict[str, Any]:
    """
    Detect available USD import parameters in Blender 4.5
    Return a dict of defaults / detected values
    """
    params = {}
    
    try:
        # Blender 4.5 operator layout
        if hasattr(bpy.ops.wm, "usd_import"):
            # Read operator RNA properties
            op_props = bpy.ops.wm.usd_import.get_rna_type().properties
            
            # Common animation / physics-related USD import flags
            anim_related_params = [
                'import_animations',        # import animations
                'read_mesh_sequences',      # mesh sequences (vertex anim)
                'import_shapes',            # shape keys (cloth deformation)
                'import_skeletons',         # skeletons
                'import_physics',           # physics (cloth, rigid body, ...)
                'import_subdiv',            # subdivision
                'set_frame_range',          # frame range
                'import_materials',         # materials
                'import_textures',          # textures
                'import_lights',            # lights
                'import_cameras',           # cameras
                'import_curves',            # curves
                'scale',                    # scale
                'light_intensity_scale',    # light intensity scale
                'relative_path',            # relative paths
                'create_collection',        # create collection
            ]
            
            for param in anim_related_params:
                if param in op_props:
                    # Pick defaults / recommended values
                    if param in ['import_animations', 'read_mesh_sequences', 
                                'import_shapes', 'set_frame_range', 'import_materials']:
                        params[param] = True
                    elif param == 'scale':
                        params[param] = 1.0
                    elif param == 'light_intensity_scale':
                        params[param] = 1.0
                    elif param == 'relative_path':
                        params[param] = False  # absolute paths (easier to relocate files)
                        
            print(f"✓ Detected {len(params)} USD import parameters for Blender 4.5")
            
    except Exception as e:
        print(f"⚠ Could not detect USD import parameters: {e}")
        # Fallback default parameter set
        params = {
            'import_animations': True,
            'read_mesh_sequences': True,
            'import_shapes': True,
            'import_skeletons': True,
            'import_physics': True,
            'set_frame_range': True,
            'import_materials': True,
            'import_textures': True,
            'relative_path': False,
            'scale': 1.0,
        }
    
    return params

def import_animated_usd(filepath: str) -> bool:
    """
    Import an animated USD (tuned for robot + cloth motion)
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"USD file not found: {filepath}")
    
    print(f"📥 Importing animated USD: {filepath}")
    
    # Start from a clean scene
    try:
        bpy.ops.wm.read_homefile(use_empty=True)
        print("✓ Scene reset to empty state")
    except Exception as e:
        print(f"⚠ Could not reset scene: {e}")
    
    # Build import kwargs
    import_params = detect_usd_import_parameters()
    import_params['filepath'] = filepath
    
    print("📋 Import parameters:")
    for key, value in import_params.items():
        if key != 'filepath':
            print(f"  {key}: {value}")
    
    try:
        # Run import
        print("🔄 Starting USD import...")
        result = bpy.ops.wm.usd_import(**import_params)
        # import pdb;pdb.set_trace()
        if 'FINISHED' in result:
            print("✓ USD import completed successfully")
            
            # Inspect imported animation
            check_imported_animation()
            
            # Optional: extra cloth physics setup
            setup_cloth_physics_if_needed()
            
            return True
        else:
            print(f"⚠ USD import returned: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error during USD import: {e}")
        traceback.print_exc()
        raise

def check_imported_animation() -> None:
    """
    Inspect imported animation data
    """
    print("\n🔍 Checking imported animation data:")
    
    scene = bpy.context.scene
    
    # Frame range
    print(f"  Frame range: {scene.frame_start} - {scene.frame_end}")
    
    # Count animation-related data
    anim_count = 0
    shapekey_count = 0
    modifier_count = 0
    
    for obj in bpy.data.objects:
        # Object actions
        if obj.animation_data and obj.animation_data.action:
            anim_count += 1
            print(f"  ✓ {obj.name} has animation action: {obj.animation_data.action.name}")
        
        # Shape keys
        if hasattr(obj.data, 'shape_keys') and obj.data.shape_keys:
            shapekey_count += 1
            key_blocks = obj.data.shape_keys.key_blocks
            print(f"  ✓ {obj.name} has {len(key_blocks)} shape keys")
        
        # Modifiers (cloth / cache / ...)
        for mod in obj.modifiers:
            if mod.type in ['CLOTH', 'SOFT_BODY', 'MESH_SEQUENCE_CACHE', 'ARMATURE']:
                modifier_count += 1
                print(f"  ✓ {obj.name} has {mod.type} modifier: {mod.name}")
    
    print(f"\n📊 Summary:")
    print(f"  Animated objects: {anim_count}")
    print(f"  Objects with shape keys: {shapekey_count}")
    print(f"  Objects with physics/cache modifiers: {modifier_count}")
    
    # Timeline markers
    if scene.timeline_markers:
        print(f"  Timeline markers: {len(scene.timeline_markers)}")

def setup_cloth_physics_if_needed() -> None:
    """
    If cloth-like meshes exist, tune cloth settings
    """
    print("\n🎭 Setting up cloth physics (if needed):")
    
    cloth_objects = []
    
    # Heuristic: name-based cloth detection
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # Naming heuristics
            cloth_keywords = ['cloth', 'fabric', 'rag', 'towel', 'sheet', 'flag']
            if any(keyword in obj.name.lower() for keyword in cloth_keywords):
                cloth_objects.append(obj)
    
    if not cloth_objects:
        print("  No cloth-like objects detected (based on naming)")
        return
    
    print(f"  Found {len(cloth_objects)} potential cloth objects")
    
    for obj in cloth_objects:
        print(f"  Setting up cloth for: {obj.name}")
        
        # Add cloth modifier if missing
        if not any(mod.type == 'CLOTH' for mod in obj.modifiers):
            cloth_mod = obj.modifiers.new(name="Cloth", type='CLOTH')
            
            # Cloth params (robot-cloth interaction)
            cloth_settings = cloth_mod.settings
            
            # Quality / mass
            cloth_settings.quality = 10
            cloth_settings.mass = 0.3  # medium-weight cloth
            
            # Simulation
            cloth_settings.air_damping = 1.0
            cloth_settings.bending_stiffness = 50.0
            
            # Collision
            cloth_settings.collision_settings.use_collision = True
            cloth_settings.collision_settings.distance_min = 0.01
            cloth_settings.collision_settings.impulse_clamp = 0.0
            
            print(f"    ✓ Added cloth modifier with basic settings")
        
        # Subdivision for simulation quality
        if obj.modifiers.get("Subdivision") is None:
            subdiv_mod = obj.modifiers.new(name="Subdivision", type='SUBSURF')
            subdiv_mod.levels = 2
            subdiv_mod.render_levels = 2
            print(f"    ✓ Added subdivision modifier for better cloth simulation")

def save_blend_with_animation(outpath: str) -> bool:
    """
    Save a .blend that keeps animation data
    """
    # Ensure output directory exists
    odir = os.path.dirname(os.path.abspath(outpath))
    if odir and not os.path.exists(odir):
        os.makedirs(odir, exist_ok=True)
    
    # Detect whether we should expect actions
    save_animation_data = False
    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            save_animation_data = True
            break
    
    if not save_animation_data:
        print("⚠ No animation data found in scene!")
    
    try:
        # Compressed .blend
        print(f"\n💾 Saving to: {outpath}")
        bpy.ops.wm.save_as_mainfile(
            filepath=outpath,
            compress=True,
            copy=True  # pack external data into .blend
        )
        
        print("✓ Blend file saved successfully")
        
        # Verify file on disk
        if os.path.exists(outpath):
            file_size = os.path.getsize(outpath) / (1024 * 1024)  # MB
            print(f"✓ File size: {file_size:.2f} MB")
            return True
        else:
            print("❌ File was not created!")
            return False
            
    except Exception as e:
        print(f"❌ Error saving blend file: {e}")
        raise

def cleanup_before_save() -> None:
    """
    Pre-save cleanup
    """
    print("\n🧹 Performing pre-save cleanup:")
    
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')
    
    # Remove unused data blocks
    for block_type in [bpy.data.meshes, bpy.data.materials, 
                      bpy.data.textures, bpy.data.images]:
        for block in block_type:
            if block.users == 0:
                block_type.remove(block)
    
    print("✓ Cleanup completed")


def step_01(usd_file, out_blend_path):
    """
    Step 1 entry: USD → .blend
    """    
    in_usd = os.path.abspath(usd_file)
    out_blend = os.path.abspath(out_blend_path)
    
    print("=" * 60)
    print(f"USD to Blend Converter (Blender 4.5 - Animation Optimized)")
    print(f"Input:  {in_usd}")
    print(f"Output: {out_blend}")
    print("=" * 60)
    
    # # Enable USD add-on
    # if not enable_usd_addon():
    #     print("❌ Failed to enable USD addon. Exiting.")
    #     return 2
    
    # Import USD
    try:
        if not import_animated_usd(in_usd):
            print("❌ USD import failed. Exiting.")
            return 3
    except Exception as e:
        print(f"❌ Exception during import: {e}")
        return 4
    
    # Cleanup
    cleanup_before_save()
    
    # Save
    try:
        if not save_blend_with_animation(out_blend):
            print("❌ Failed to save blend file.")
            return 5
    except Exception as e:
        print(f"❌ Exception during save: {e}")
        return 6
    
    print("\n" + "=" * 60)
    print("✅ Conversion completed successfully!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    # CLI args after --
    args = sys.argv
    
    if "--" in args:
        args = args[args.index("--") + 1:]
    else:
        # No -- : use last two argv tokens
        args = args[-2:] if len(args) >= 2 else []
    
    exit_code = step_01(args[0], args[1])
    sys.exit(exit_code)