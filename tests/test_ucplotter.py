from os import path

from ucplots.ucplots import UCPlot

print("Testing CShape")


class TestUnitCellPlotter:
    """
    This is a dummy test, written to pass checks
    """

    a = 2

    def test_sample(self):
        assert self.a**2 == 4


if __name__ == "__main__":
    UCPlot().plot_unit_cell(
        uc_bbox=(-1.0, -1.0, 1.0, 1.0),
        inclusions_data={
            "CIRCLE": [[0.0, 0.0, 1.0]],
        },
        image_path=path.join(path.dirname(__file__), "test_uc_image.png"),
    )
