from pkg import *

# 水动力计算
def hydro(body,omega,depth,wave_direction,HTM=False,save = False):
    problems = [cpt.RadiationProblem(body=body, omega=omega,rho=1025,sea_bottom=depth,radiating_dof=dof) for dof in body.dofs]
    problems += [cpt.DiffractionProblem(omega=omega, body=body,rho=1025,wave_direction=wave_direction,sea_bottom=depth)]
    
    bem_solver = cpt.BEMSolver()
    # 判断是否使用稀疏矩阵
    if HTM == True:
        sparse_engine = cpt.HierarchicalToeplitzMatrixEngine()
        sparse_solver = cpt.BEMSolver(engine=sparse_engine)
        start_time = time.perf_counter()#运行时间
        result = sparse_solver.solve_all(problems)
        dataset = cpt.assemble_dataset(result) 
    start_time = time.perf_counter()#运行时间
    result = bem_solver.solve_all(problems)
    dataset = cpt.assemble_dataset(result,wavelength=True) 
    print("Dense resolution time: ", time.perf_counter() - start_time, "seconds")
    # set beem element stiffness and computing rao
    if save == True:
        cpt.io.xarray.separate_complex_values(dataset).to_netcdf(f'BM8_120.nc',
                    encoding={'radiating_dof': {'dtype': 'U'},
                                'influenced_dof': {'dtype': 'U'}})
    return dataset
