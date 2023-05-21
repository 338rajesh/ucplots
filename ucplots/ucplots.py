import io
from os import path

import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca, figure, Axes, xlim, ylim, axis, savefig, clf
from numpy import frombuffer, uint8, reshape, transpose
from numpy import save, load, array, ndarray
from PIL import Image

from .plot_2D_shapes import Plot2DShapes
from .utils import ProgressBar


class UCPlot:
    def __init__(self):
        return

    valid_fiber_shapes = (
        'CSHAPE', 'CAPSULE', 'CIRCLE', 'ELLIPSE', 'NLOBESHAPE', 'N_TIP_STAR', 'RECTANGLE', 'REGULARPOLYGON',
    )

    default_kwargs = {
        "image_extension": "png",
        "matrix_color": "0",
        "fibre_color": "1",
        "matrix_edge_color": None,
        "fibre_edge_color": None,
        "fibre_edge_thickness": None,
        "angle_units": 'radians',
        "z_comp_in_data": False,
        "pixels": (256, 256),
        "get_image_array": False,
        "image_mode": "L",
        "verbose": 1,
        "dither": False,
    }
    """  
      default keyword arguments
      =========================
        + `image_extension`: str "png"
        + "matrix_color": "black"
        + "fiber_color": "white"
        + "matrix_edge_color": None
        + "fibre_edge_color" : None
        + "fibre_edge_thickness": None
        + "angle_units" : 'radians'
        + "z_comp_in_data" : False
        + "pixels" : (256, 256)
        + "get_image_array" : False
        + "img_mode": "L"  # L-Gray scale, 1-Binary, P-
        + "verbose" : 1
    """

    def _set_default_plot_options(self, **user_kwargs):
        for (dkw_arg, dkw_val) in self.default_kwargs.items():
            if dkw_arg not in user_kwargs.keys():
                user_kwargs[dkw_arg] = dkw_val
        return user_kwargs

    def _assert_fiber_shapes_validity(self, f_shapes: list[str]):
        for af_shape in f_shapes:
            assert af_shape.upper() in self.valid_fiber_shapes, (
                f"Found invalid fiber shape {af_shape} while valid shapes are {self.valid_fiber_shapes}"
            )

    @staticmethod
    def _get_image_array(_fig):
        io_buffer = io.BytesIO()
        savefig(io_buffer, format="raw")
        io_buffer.seek(0)
        _image_array = reshape(
            frombuffer(io_buffer.getvalue(), dtype=uint8),
            newshape=(int(_fig.bbox.bounds[3]), int(_fig.bbox.bounds[2]), -1)
        )
        io_buffer.close()
        return _image_array

    def plot_unit_cell(self,
                       uc_bbox: tuple | list,
                       inclusions_data,
                       image_path: str = None,
                       **kwargs
                       ):
        """

        It **plots a single unit cell** having bounding box `uc_bbox` and inclusions/ fibers data given by
        a dictionary `inclusions_data`.

        Parameters
        ----------
        uc_bbox
        inclusions_data
        image_path

        Returns
        -------

        """

        kwargs = self._set_default_plot_options(**kwargs)
        figsize_pixels = (kwargs["pixels"][0] * 0.01, kwargs["pixels"][1] * 0.01)
        fig = figure(0, figsize=figsize_pixels, frameon=False)
        ax = Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        #
        # plot RUC
        Plot2DShapes.plot_bbox(bounds=uc_bbox,
                               fig_handle=gca(),
                               fc= kwargs['matrix_color'],
                               ec= kwargs['matrix_edge_color'], 
                            )
        # plot inclusions
        self._assert_fiber_shapes_validity(list(inclusions_data.keys()))
        fib_plot_kwargs = {
            "ec": kwargs['fibre_edge_color'],
            "fc": kwargs['fibre_color'],
            "et": kwargs['fibre_edge_thickness']
        }
        for (fibres_shape, inc_data) in inclusions_data.items():
            if fibres_shape.upper() == "CIRCLE":
                Plot2DShapes.plot_circular_discs(xyr=inc_data, fig_handle=gca(), **fib_plot_kwargs, )
            elif fibres_shape.upper() == "CAPSULE":
                Plot2DShapes.plot_capsular_discs(xyt_ab=inc_data, fig_handle=gca(), **fib_plot_kwargs, )
            elif fibres_shape.upper() == "ELLIPSE":
                Plot2DShapes.plot_elliptical_discs(xyt_ab=inc_data, fig_handle=gca(), **fib_plot_kwargs, )
            elif fibres_shape.upper() == "RECTANGLE":
                Plot2DShapes.plot_rectangles(xyt_abr=inc_data, fig_handle=gca(), **fib_plot_kwargs, )
            elif fibres_shape.upper() == "REGULARPOLYGON":
                Plot2DShapes.plot_regular_polygons(xyt_a_rf_n=inc_data, fig_handle=gca(), **fib_plot_kwargs, )
            elif fibres_shape.upper() == "NLOBESHAPE":
                Plot2DShapes.plot_nlobe_shapes(xyt_ro_rl_n=inc_data, fig_handle=gca(), **fib_plot_kwargs)
            elif fibres_shape.upper().startswith("N_TIP_STAR"):
                Plot2DShapes.plot_stars(xyt_ro_rb_rtf_rbf_n=inc_data, fig_handle=gca(), **fib_plot_kwargs)
            elif fibres_shape.upper() == "CSHAPE":
                Plot2DShapes.plot_cshapes(xyt_ro_ri_alpha=inc_data, fig_handle=gca(), **fib_plot_kwargs)

        # display or save
        axis("off")
        xlim([uc_bbox[0], uc_bbox[2]])
        ylim([uc_bbox[1], uc_bbox[3]])
        image_array = self._get_image_array(fig)

        if kwargs['image_mode'] in ('L', '1'):
            Image.fromarray(image_array).convert(mode=kwargs['image_mode'], dither=kwargs['dither']).save(image_path)
        else:
            plt.savefig(image_path)
        clf()
        #
        return fig, image_array

    def unit_cells_from_h5(
            self,
            h5file: str,
            images_dir: str = None,
            npy_path: str = None,
            **kwargs,
    ):
        """

        Plots unit cells whose data is given as .h5 file with the following structure,

        -root-
            |-- a_unit_cell-1
                            |-- shape_1 data_set
                            |-- shape_2 data_set
                            |-- shape_3 data-set
                            |-- .
                            |-- .
                            |-- shape_n data-set
            |-- a_unit_cell-2
                            |-- shape_1 data_set
                            |-- shape_2 data_set
                            |-- shape_3 data-set
                            |-- .
                            |-- .
                            |-- shape_n data-set

        Parameters
        ----------
        npy_path
        h5file
        images_dir
        kwargs

        Returns
        -------

        """
        #
        kwargs = self._set_default_plot_options(**kwargs)
        with h5py.File(h5file, "r") as h5fid:
            plt_image_path = None
            image_arrays = []
            if images_dir is None:
                _pbar_header = f"Making unit cells data set in numpy array format"
            else:
                _pbar_header = f"Plotting unit cell images at {images_dir}"
            p_bar = ProgressBar(len(h5fid.keys()), header=_pbar_header)
            for (i, (dsID, ds)) in enumerate(h5fid.items()):
                if images_dir is not None:
                    image_name_suffix = str(dsID)
                    plt_image_path = path.join(images_dir, f"{image_name_suffix}.{kwargs['image_extension']}")
                #
                data = {ak: transpose(av) for (ak, av) in ds.items()}
                img = self.plot_unit_cell(
                    uc_bbox=(ds.attrs["xlb"], ds.attrs["ylb"], ds.attrs["xub"], ds.attrs["yub"],),
                    inclusions_data=data,
                    image_path=plt_image_path,
                    **kwargs
                )
                #
                if img is not None and isinstance(img, ndarray):
                    image_arrays.append(img)
                p_bar.update(i)
            #
            if len(image_arrays) > 0:
                if npy_path is None:
                    return image_arrays
                else:
                    save(npy_path, array(image_arrays))

    def plot_unit_cell_from_dict(
            self,
            data: dict,
            images_dir: str,
            image_name: str,
            **kwargs,
    ):
        """
        Plots **a single unit cell** from npz file which contains bounding box array with 'bbox' key and
        remaining key-value pairs *should be* inclusion shape name and data pairs.

        Parameters
        ----------
        data
        images_dir
        image_name
        kwargs

        Returns
        -------

        """
        uc_bbox = tuple(data["bbox"].ravel())
        data.pop("bbox")
        self.plot_unit_cell(
            uc_bbox=uc_bbox,
            inclusions_data=data,
            image_path=path.join(images_dir, f"{image_name}.{kwargs['image_extension'].lower()}"),
            **kwargs
        )
        return

    def plot_unit_cell_from_npz(
            self,
            npz_file: str,
            images_dir: str,
            image_name: str,
            **kwargs,
    ):
        """
        Plots **a single unit cell** from npz file which contains bounding box array with 'bbox' key and
        remaining key-value pairs *should be* inclusion shape name and data pairs.

        Parameters
        ----------
        npz_file
        images_dir
        image_name
        kwargs

        Returns
        -------

        """
        return self.plot_unit_cell_from_dict(
            dict(load(npz_file)),
            images_dir,
            image_name,
            **kwargs,
        )
