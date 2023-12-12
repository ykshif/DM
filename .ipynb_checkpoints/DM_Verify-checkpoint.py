import matplotlib.pyplot as plt
import xarray as xr
from capytaine.io.xarray import merge_complex_values
from capytaine.post_pro import rao

def verify_hydrodynamic_data(filepath, wave_direction=0.0, dissipation=None, stiffness=None):
    """
    Verify the hydrodynamic data and calculate the rao for a given wave direction.

    Parameters:
    - filepath (str): Path to the .nc file containing the dataset.
    - wave_direction (float, default=0.0): Direction of the wave.
    - dissipation (type?, default=None): Dissipation parameter. Define its type if known.
    - stiffness (type?, default=None): Stiffness parameter. Define its type if known.

    Returns:
    - dataset (xarray.Dataset): Dataset after processing and addition of rao values.
    """

    # Load the dataset from the given file and merge its complex values
    dataset = merge_complex_values(xr.open_dataset(filepath))
    
    # Calculate the rao and add it to the dataset
    dataset['rao'] = rao(dataset, wave_direction=wave_direction, dissipation=dissipation, stiffness=stiffness)
    
    # Plot the rao
    plt.plot(abs(dataset['rao'][0][2::6]),marker='o')
    plt.xlabel('Frequency Index')  # You can adjust this label if needed
    plt.ylabel('RAO Magnitude')  # You can adjust this label if needed
    plt.title('RAO for Wave Direction {}'.format(wave_direction))
    plt.show()

    return dataset

# # Example usage:
# dataset_result = verify_hydrodynamic_data("BM10_180_direction0.nc")
