import bpy


for lamp in bpy.data.lamps:
    bpy.data.lamps.remove(lamp) 

#bpy.ops.object.mode_set(mode='OBJECT')
#bpy.ops.object.select_by_type(type='MESH')
#bpy.ops.object.delete(use_global=False) 



for item in bpy.data.meshes:
    bpy.data.meshes.remove(item)
cubes = []
mats = [] 
mat = bpy.data.materials.new(name = 'red')
mat.diffuse_color = (1., 0., 0.)
mats.append(mat)

mat = bpy.data.materials.new(name = 'green')
mat.diffuse_color = (0., 1., 0.)
mats.append(mat)

mat = bpy.data.materials.new(name = 'red')
mat.diffuse_color = (0., 0., 1.)
mats.append(mat)

groups = bpy.data.groups
group = groups.get("cubes", groups.new("cubes"))

for j in range(3):
    for i in range(20):
        bpy.ops.mesh.primitive_cube_add(location = (j,i,i))
        cube = bpy.context.object
        bpy.ops.transform.resize(value=(.45, .45, .45))

        # Assign the material to the cube.
        mesh = cube.data
        mesh.materials.append(mats[j])
        cubes.append(cube)
        cube.select = True
        #group.objects.link(cube)
       

for obj in bpy.data.objects:
   obj.select = True

bpy.context.scene.objects.active = bpy.data.objects['Cube.030']

bpy.ops.object.join()

bpy.ops.object.lamp_add(type='POINT', view_align=False, location=(10, 10, 15))
bpy.ops.object.lamp_add(type='POINT', view_align=False, location=(10, -10, 15))
bpy.ops.object.lamp_add(type='POINT', view_align=False, location=(-10, 10, 15) )
bpy.ops.object.lamp_add(type='POINT', view_align=False, location=(-10, -10, 15) )

