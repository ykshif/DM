from pkg import *

# 结构模型
def Create_geometry(x,y,z,m,h,nx_distance,ny_distance,nx,ny,dof=6,mesh=1.5,save = False):
    #1、建立几何模型整列
    #2、x,y,z为长宽高，m为质量，h为吃水深度
    #3、dof为自由度，0为6自由度，1为Heave
    #4、nx_distance整行的间距，x轴方向,ny_distance整列的间距，y轴方向
    #5、nx为行数，ny为列数
    #6、网格尺寸为mesh=1.5 
    #7、save为是否保存stl文件,为wecsim准备
    body = cpt.RectangularParallelepiped(size=(x, y, z), resolution= (int(x/mesh),int(y/mesh),int(z/mesh)), center=(0, 0, h))
    body.center_of_mass = (0,0,h)
    body.mass = m
    if dof == 2:
        body.add_translation_dof(name="Heave")
        body.add_rotation_dof(name="Pitch")
    else:
        body.add_all_rigid_body_dofs()
    body.inertia_matrix = body.compute_rigid_body_inertia()
    body.keep_immersed_part(free_surface=0)
    body.rotation_center = body.center_of_mass
    body.hydrostatic_stiffness = body.compute_hydrostatic_stiffness()
    body.name = f"body_{body.mesh.nb_faces:04d}"
    body.show_matplotlib()
    array_of_rec = body.assemble_regular_array(nx_distance,(nx,1))
    if ny > 1:
        array_of_rec = array_of_rec.assemble_regular_array(ny_distance,(1,ny))
    array_of_rec.inertia_matrix = array_of_rec.add_dofs_labels_to_matrix(block_diag(*[body.inertia_matrix for _ in range(nx*ny)]))
    array_of_rec.hydrostatic_stiffness= array_of_rec.add_dofs_labels_to_matrix(block_diag(*[body.hydrostatic_stiffness for _ in range(nx*ny)]))
    array_of_rec.show_matplotlib()
    #保存模型为stl文件
    if save == True:
        write_STL("array_of_rec.stl",array_of_rec.mesh.vertices,array_of_rec.mesh.faces)
    return array_of_rec