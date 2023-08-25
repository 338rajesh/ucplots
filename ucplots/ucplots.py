import io
from multiprocessing import Pool
from os import path

from PIL.Image import fromarray
from h5py import File
from matplotlib.pyplot import gca, figure, Axes, xlim, ylim, axis, savefig, clf
from numpy import frombuffer, uint8, reshape, transpose
from tqdm import tqdm

from .plot_2D_shapes import Plot2DShapes


class UCPlot:
    """
    Unit Cell Plotter

    """

    def __init__(
        self,
        image_extension: str = "png",
        matrix_color: str = "black",
        fibre_color: str = "white",
        matrix_edge_color: str = None,
        fibre_edge_color: str = None,
        fibre_edge_thickness: float = None,
        pixels: tuple[int, int] | int = (256, 256),
        image_mode: str = "L",  # L-Gray scale, 1-Binary, see PIL.Image documentation for details
        dither: bool = 0,
    ):
        """

        Parameters
        ----------
        image_extension: str, All the extension supported by `savefig` function of `matplotlib.pyplot`.
        matrix_color: str
        fibre_color: str
        matrix_edge_color: str
        fibre_edge_color: str
        fibre_edge_thickness: float
        pixels: int | tuple[int, int]
        image_mode: str
        dither: bool
        """
        self.image_extension = image_extension
        self.matrix_color = matrix_color
        self.fibre_color = fibre_color
        self.matrix_edge_color = matrix_edge_color
        self.fibre_edge_color = fibre_edge_color
        self.fibre_edge_thickness = fibre_edge_thickness
        if isinstance(pixels, int) or isinstance(pixels, float):
            pixels = (int(pixels), int(pixels))
        self.pixels = pixels
        self.image_mode = image_mode
        self.dither = dither
        return

    @staticmethod
    def valid_fiber_shapes():
        """

        + 'CSHAPE',
        + 'CAPSULE',
        + 'CIRCLE',
        + 'ELLIPSE',
        + 'NLOBESHAPE',
        + 'N_TIP_STAR',
        + 'RECTANGLE',
        + 'REGULARPOLYGON',

        """
        return (
            "CSHAPE",
            "CAPSULE",
            "CIRCLE",
            "ELLIPSE",
            "NLOBESHAPE",
            "N_TIP_STAR",
            "RECTANGLE",
            "REGULARPOLYGON",
        )

    def _assert_fiber_shapes_validity(self, f_shapes: list[str]):
        for af_shape in f_shapes:
            assert (
                af_shape.upper() in self.valid_fiber_shapes()
            ), f"Found invalid fiber shape {af_shape} while valid shapes are {self.valid_fiber_shapes()}"

    @staticmethod
    def _get_image_array(_fig):
        io_buffer = io.BytesIO()
        savefig(io_buffer, format="raw")
        io_buffer.seek(0)
        _image_array = reshape(
            frombuffer(io_buffer.getvalue(), dtype=uint8),
            newshape=(int(_fig.bbox.bounds[3]), int(_fig.bbox.bounds[2]), -1),
        )
        io_buffer.close()
        return _image_array

    def plot_unit_cell(
        self,
        uc_bbox: tuple | list,
        inclusions_data: dict,
        image_path: str = None,
    ):
        """**Plots a single unit cell**

        For this purpose, one needs to supply  `uc_bbox`, bounding box as list/tuple of four floats
         and `inclusions_data` as a dictionary with str-array pairs.

        Parameters
        ----------
        uc_bbox
        inclusions_data
        image_path

        Returns
        -------

        """

        fig = figure(
            num=0, figsize=(self.pixels[0] * 0.01, self.pixels[1] * 0.01), frameon=False
        )
        ax = Axes(fig, rect=[0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax)
        #
        # plot RUC bounds
        Plot2DShapes.plot_bbox(
            bounds=uc_bbox,
            fig_handle=gca(),
            fc=self.matrix_color,
            ec=self.matrix_edge_color,
        )
        # plot inclusions
        self._assert_fiber_shapes_validity(list(inclusions_data.keys()))
        for (fibres_shape, inc_data) in inclusions_data.items():
            if fibres_shape.upper() == "CIRCLE":
                plot_func = Plot2DShapes.plot_circular_discs
            elif fibres_shape.upper() == "CAPSULE":
                plot_func = Plot2DShapes.plot_capsular_discs
            elif fibres_shape.upper() == "ELLIPSE":
                plot_func = Plot2DShapes.plot_elliptical_discs
            elif fibres_shape.upper() == "RECTANGLE":
                plot_func = Plot2DShapes.plot_rectangles
            elif fibres_shape.upper() == "REGULARPOLYGON":
                plot_func = Plot2DShapes.plot_regular_polygons
            elif fibres_shape.upper() == "NLOBESHAPE":
                plot_func = Plot2DShapes.plot_nlobe_shapes
            elif fibres_shape.upper().startswith("N_TIP_STAR"):
                plot_func = Plot2DShapes.plot_stars
            elif fibres_shape.upper() == "CSHAPE":
                plot_func = Plot2DShapes.plot_cshapes
            else:
                raise Warning(f"Invalid fibre shape: {fibres_shape.upper()}")
            #
            plot_func(
                gca(),
                inc_data,
                ec=self.fibre_edge_color,
                fc=self.fibre_color,
                et=self.fibre_edge_thickness,
            )
        # display or save
        axis("off")
        xlim([uc_bbox[0], uc_bbox[2]])
        ylim([uc_bbox[1], uc_bbox[3]])
        image_array = self._get_image_array(fig)

        if self.image_mode in ("L", "1"):
            fromarray(image_array).convert(
                mode=self.image_mode, dither=self.dither
            ).save(image_path)
        else:
            savefig(image_path)
        clf()
        #
        return fig, image_array

    def _plot_uc_(self, args: tuple):
        """Private function to support parallel plotting with tqdm"""
        self.plot_unit_cell(*args)

    def plot_unit_cells(
        self,
        file_path,
        images_dir=None,
        num_cores=None,
    ):
        """**Plots multiple unit cells**

        Built on top of `plot_unit_cell()`, it plots multiple unit cells with data obtained from
        the file located at `file_path`.

        At present, the acceptable file formats are

            + `*.h5` with the following structure. A single file can hold the information for `n` different unit
            cells as

                + ROOT
                    * a_unit_cell-1
                        + shape_1 data_set
                        + shape_2 data_set
                        + .
                        + shape_n data-set
                    * a_unit_cell-2
                        + shape_1 data_set
                        + shape_2 data_set
                        + .
                        + shape_n data-set

           + `*.npz` with inclusion shape ids as keys and their respective positional and shape information as values



        Parameters
        ----------
        file_path
        images_dir
        num_cores

        Returns
        -------

        """
        assert path.isfile(
            file_path
        ), f"The specified file path {file_path} is not valid."
        if file_path.split(".")[-1].lower() == "h5":
            h5fid = File(file_path, mode="r")
            data_items = list(h5fid.items())
            if num_cores is None:
                p_bar = tqdm(
                    ascii=True, total=len(data_items), desc="Plotting Unit Cells"
                )
                for (i, (dsID, ds)) in enumerate(data_items):
                    self.plot_unit_cell(
                        uc_bbox=(
                            ds.attrs["xlb"],
                            ds.attrs["ylb"],
                            ds.attrs["xub"],
                            ds.attrs["yub"],
                        ),
                        inclusions_data={ak: transpose(av) for (ak, av) in ds.items()},
                        image_path=path.join(
                            images_dir, f"{str(dsID)}.{self.image_extension}"
                        ),
                    )
                    p_bar.update(1)
                p_bar.close()
            else:
                assert isinstance(num_cores, int), "number of cores must be a integer"
                pool = Pool(processes=num_cores)
                pool.imap(
                    self._plot_uc_,
                    tqdm(
                        [
                            (
                                (
                                    ds.attrs["xlb"],
                                    ds.attrs["ylb"],
                                    ds.attrs["xub"],
                                    ds.attrs["yub"],
                                ),
                                {ak: transpose(av) for (ak, av) in ds.items()},
                                path.join(
                                    images_dir, f"{str(ds_id)}.{self.image_extension}"
                                ),
                            )
                            for (ds_id, ds) in data_items
                        ],
                        total=len(data_items),
                    ),
                )
                #
                pool.close()
                pool.join()
            #
            h5fid.close()
        else:
            raise ValueError(f"Invalid data source!")
