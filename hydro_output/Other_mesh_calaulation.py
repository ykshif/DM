
from pkg import *
# 读取第三方网格开展计算
mesh = cpt.io.mesh_loaders.load_mesh('E:\phd\Code\DM-FEM2D\hydro_output\sphere.dat', name=None)
body = cpt.FloatingBody(mesh=mesh, center_of_mass=(0,0,-2))
body.add_all_rigid_body_dofs()
body.keep_immersed_part()
body.show_matplotlib()

body.inertia_matrix = body.compute_rigid_body_inertia()
body.hydrostatic_stiffness = body.compute_hydrostatic_stiffness()
omega = np.linspace(0.01, 8, 420)
bem_solver = cpt.BEMSolver()
problems = [cpt.RadiationProblem(body=body, omega=o,rho=1025,sea_bottom=-50,radiating_dof=dof) for o in omega for dof in body.dofs]
problems += [cpt.DiffractionProblem(omega=o, body=body,rho=1025,wave_direction=0,sea_bottom=-50) for o in omega]
result = bem_solver.solve_all(problems,n_jobs=20)
cpt.io.xarray.separate_complex_values(dataset).to_netcdf(f'self_sphere.nc',
            encoding={'radiating_dof': {'dtype': 'U'},
                        'influenced_dof': {'dtype': 'U'}})

