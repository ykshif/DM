import numpy as np
import matplotlib.pyplot as plt

class WindLoad:
    def __init__(self, U10, z, file_path, A=1, total_rows=10, total_cols=50, rho=1.225, alpha=0.125):
        """
        Initialize the WindLoad class with the given parameters.
        
        Parameters:
        - U10: Wind speed at 10m above ground level.
        - z: Height above ground level.
        - file_path: Path to the scatter data file.
        - A: Area (default is 1) projection.
        - total_rows: Number of rows for wind load coefficient matrix.
        - total_cols: Number of columns for wind load coefficient matrix.
        - rho: Air density (default is 1.225 kg/m^3).
        - alpha: Wind profile power law exponent (default is 0.125).
        """
        self.U10 = U10
        self.z = z
        self.file_path = file_path
        self.A = A
        self.total_rows = total_rows
        self.total_cols = total_cols
        self.rho = rho
        self.alpha = alpha

    def adjust_wind_speed(self):
        """
        Compute and return the adjusted wind speed based on height.
        Adjusts the wind speed using the power law.
        """
        return self.U10 * (self.z / 10)**self.alpha

    def turbulence_intensity(self):
        """
        Compute and return the turbulence intensity based on height.
        Calculates turbulence intensity using a piecewise power law.
        """
        exponent = -0.125 if self.z <= 20 else -0.275
        return 0.15 * (self.z / 20)**exponent

    def api_spectrum(self):
        """
        Compute and return the API wind spectrum.
        Calculates the wind spectrum based on API standards.
        """
        f = np.linspace(0.01, 2.0, 100)
        U = self.adjust_wind_speed()
        Ti = self.turbulence_intensity()
        fp_value = 0.025 * U / self.z
        return U**2 * Ti**2 / fp_value * (1 + 1.5 * (f / fp_value))**(-5 / 3)

    def compute_amplitude_from_spectrum(self):
        """
        Compute and return the amplitude values from the wind spectrum.
        Calculates the amplitude using the square root of the spectrum.
        """
        spectrum = self.api_spectrum()
        delta_omega = 0.05
        return np.sqrt(2 * spectrum * delta_omega)

    def compute_amplitude_for_frequency(self, target_frequency):
        """
        Compute and return the amplitude for a given frequency.
        
        Parameters:
        - target_frequency: The desired frequency for which amplitude is computed.
        """
        amplitude = self.compute_amplitude_from_spectrum()
        f = np.linspace(0.01, 2.0, 100)
        index = np.abs(f - target_frequency).argmin()
        # Set a random seed for reproducibility
        np.random.seed(0)
        # Generate random phase for each frequency
        phi = np.random.uniform(0, 2*np.pi, 100)
        # Convert amplitude to complex form using the generated phase
        amplitude = amplitude * (np.cos(phi) + 1j * np.sin(phi))
        
        return amplitude[index]

    def wind_load_coefficient(self):
        """
        Compute and return the wind load coefficient based on scatter data.
        Uses scatter data from a file to compute the wind load coefficient.
        """
        
        def read_scatter_data():
            """Helper function to read the scatter data from the class file_path."""
            data = np.loadtxt(self.file_path)
            return data[:, 0], data[:, 1]

        def extend_last_value(x_values, y_values):
            """Helper function to extend the last y value to the total_cols."""
            extended_x_values = np.arange(1, self.total_cols + 1)
            extended_y_values = np.concatenate([y_values, [y_values[-1]] * (self.total_cols - len(y_values))])
            return extended_x_values, extended_y_values

        x_data, y_data = read_scatter_data()
        _, extended_y = extend_last_value(x_data, y_data)
        return np.tile(extended_y, (self.total_rows, 1))

    def compute_wind_force(self, target_frequency,dof=0):
        """
        Compute and return the wind force for a given frequency.
        
        Parameters:
        - target_frequency: The desired frequency for which wind force is computed.
        """
        amplitude = self.compute_amplitude_for_frequency(target_frequency)
        Cd = self.wind_load_coefficient()
        V_avg = self.adjust_wind_speed()
        #每个方向上的力
        wind_force_in_one_dof = 2 * Cd * V_avg * amplitude * self.A * self.rho
        #插入完整矩阵当中
        wind_force_in_one_dof = wind_force_in_one_dof.reshape(self.total_rows*self.total_cols)
        # Complete node force matrix
        force_matrix = np.zeros((1,self.total_rows*self.total_cols*6),dtype=complex)
        force_matrix[0,dof::6] = wind_force_in_one_dof

        return force_matrix

    def compute_wind_damping(self,dof=0):
        """
        Compute and return the wind damping.
        Calculates the wind damping based on wind speed and coefficient.
        """
        V_avg = self.adjust_wind_speed()
        Cd = self.wind_load_coefficient()
        wind_damping = 2 * Cd * V_avg * self.A * self.rho
        wind_damping = wind_damping.reshape(self.total_rows*self.total_cols)
        global_damping = np.zeros(self.total_rows*self.total_cols*6)
        # 切片，将矩阵插入其中
        global_damping[dof::6] = wind_damping
        # 将阻尼放在对角线位置
        # damping = np.diag(global_damping)
        return global_damping


# Example usage
# wind_load = WindLoad(U10=14.3, z=2, file_path="winddata/Ti0.1degree0.txt")
# y = wind_load.wind_load_coefficient()
# amplitude_spectrum = wind_load.compute_amplitude_from_spectrum()
# spectrum = wind_load.api_spectrum()
# plt.plot(spectrum)
# plt.show()