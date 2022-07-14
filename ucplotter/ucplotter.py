from numpy import amax, amin, asarray, uint8
from PIL import Image


class TwoPhase_RVEImage:
    """
    Two Phase RVE image

    It implements all the methods related to
    unit cell plots and its manipulations using
    built in PIL module's Image sub-module.


    """

    def __init__(self, img_path=None):
        super(TwoPhase_RVEImage, self).__init__()
        if img_path is not None:
            self.img: Image.Image = Image.open(img_path)

    def resize(self, req_size: tuple, resampling_method=Image.BICUBIC):
        return self.img.resize(req_size, resample=resampling_method)

    def embed_info(
        self,
        fiber_value: float,
        matrix_value: float,
        init_value: float,
        end_value: float,
    ) -> Image.Image:
        """
        It embeds necessary information into the image based on pixel value.


        """
        # assert end_value >= fiber_value >= init_value, f"fiber properties are out of bounds, it must be between {init_value} and {end_value}"
        # assert end_value >= matrix_value >= init_value, f"matrix properties are out of bounds, it must be between {init_value} and {end_value}"
        f_pxv = ((fiber_value - init_value) / (end_value - init_value)) * 255.0
        m_pxv = ((matrix_value - init_value) / (end_value - init_value)) * 255.0

        if self.img.mode == "L":
            aimg = asarray(self.img) / 255.0
        assert (amin(aimg) >= 0.0) and (
            amax(aimg) <= 1.0
        ), "Image pixel values are outside the bounds of (0.0, 1.0)"

        return Image.fromarray((m_pxv + (aimg * (f_pxv - m_pxv))).astype(uint8))
