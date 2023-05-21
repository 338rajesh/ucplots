from numpy import pi
from ucplots.ucplots import UCPlot
from os import path

print("Testing CShape")
ucp = UCPlot()
ucp.plot_unit_cell(
    uc_bbox=(-10.0, -10.0, 10.0, 10.0),
    inclusions_data={"C": [
        [1.0, 1.0, 0.0 * pi, 6.0, 3.0, 0.5 * pi, ]], },
    image_path=path.join(path.expanduser("~"), "test_c_shape.png"),
    fibre_color="0.1",
    matrix_color="0.6",
)
