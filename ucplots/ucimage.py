from numpy import amax, amin, asarray, uint8, ndarray, zeros, dstack
from PIL import Image
from .utils import ProgressBar


class TwoPhaseUCImage:
    """
    Two Phase RVE image

    It implements all the methods related to
    unit cell plots and its manipulations using
    built in PIL module's Image submodule.


    """

    def __init__(self, img_path=None, img=None,):
        super(TwoPhaseUCImage, self).__init__()
        self.img: Image.Image = Image.open(img_path) if img_path is not None else None
        self.img: Image.Image = img if (isinstance(img, Image.Image) and self.img is None) else None

    def resize(self, req_size: tuple, resampling_method=Image.BICUBIC):
        self.img = self.img.resize(req_size, resample=resampling_method)

    def embed_data(
            self,
            fiber_value: float,
            matrix_value: float,
            data_range: tuple[float, float],
            image_path=None,
            image_array=None,
            inp_image_mode="L",
    ) -> Image.Image:
        """
        It embeds necessary information into the image based on pixel value.


        """
        # assert end_value >= fiber_value >= init_value, f"fiber properties are out of bounds, it must be between {
        # init_value} and {end_value}" assert end_value >= matrix_value >= init_value, f"matrix properties are out of
        # bounds, it must be between {init_value} and {end_value}"
        data_start, data_end = data_range
        f_pxv = ((fiber_value - data_start) / (data_end - data_start)) * 255.0
        m_pxv = ((matrix_value - data_start) / (data_end - data_start)) * 255.0

        if image_array is None:
            if image_path is None:
                assert self.img is not None, "Image doesnt exist!"
            else:
                self.img = Image.open(image_path)
            #
            if self.img.mode == "L":
                image_array = asarray(self.img) / 255
            elif self.img.mode == "1":
                image_array = asarray(self.img)
            else:
                raise TypeError("\nAt present, only BW and grayscale images are handled," +
                                f" found {self.img.mode} type image.")
        else:
            if inp_image_mode == "L":
                image_array = image_array / 255

        assert (amin(image_array) >= 0.0) and (amax(image_array) <= 1.0), (
            "Image pixel values are outside the bounds of (0.0, 1.0)"
        )
        #
        image_array = (m_pxv + (image_array * (f_pxv - m_pxv))).astype(uint8)
        #
        assert (amin(image_array) >= 0) and (amax(image_array) <= 255), (
            "Image pixel values are outside the bounds of (0, 255)"
        )
        #
        return Image.fromarray(image_array, mode="L")


def encode_material_info(images: ndarray, *properties):
    """

    Parameters
    ----------
    images
    properties: tuple of nd_arrays with each array as same number of columns as number of phases.
    Number of arrays should be equal to the number of properties in the microstructure.

    Returns
    -------

    """
    num_channels = len(properties)
    mp_extreme_values = [(amin(a_property), amax(a_property)) for a_property in properties]
    #
    p_bar = ProgressBar(
        images.shape[0],
        header=f"Encoding material info in {num_channels} channels for {images.shape[1:]} shaped images."
    )
    images_with_material_info = zeros(shape=(*images.shape[:-1], num_channels), dtype=images.dtype)
    for (i, a_image) in enumerate(images):
        uci = TwoPhaseUCImage()
        images_with_material_info[i, :, :, :] = dstack(
            [uci.embed_data(
                *properties[j][i], data_range=mp_extreme_values[j], image_array=a_image[:, :, 0]
            ) for j in range(num_channels)]
        )
        p_bar.update(i)
    return images_with_material_info
