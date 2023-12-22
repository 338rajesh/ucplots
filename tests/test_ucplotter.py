from ucplots.ucplots import UCPlot
from os import path

print("Testing CShape")

if __name__ == '__main__':
    UCPlot().plot_unit_cell(
        uc_bbox=(-1.0, -1.0, 1.0, 1.0),
        inclusions_data={"CIRCLE": [[0.0, 0.0, 1.0]], },
        image_path=path.join(path.dirname(__file__), "test_uc_image.png")
    )
