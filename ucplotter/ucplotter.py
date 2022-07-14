from .plot_2D_shapes import Plot2DShapes
from PIL import Image
from numpy import frombuffer, uint8, amin, amax, array, reshape, pi, transpose
from matplotlib.pyplot import gca, figure, Axes, xlim, ylim, axis, savefig, clf
import io
import h5py
from os import path


default_kwargs = {
    "image_extn": ".png",
    "matrix_color": "black",
    "fibre_color": "white",
    "matrix_edge_color": None,
    "fibre_edge_color": None,
    "fibre_edge_thickness": None,
    "angle_units": 'radians',
    "z_comp_in_data": False,
    "pixels": (256, 256),
    "get_imarray": False,
    "img_type": "BINARY_SCALE",
    "verbose": 1,
}
""" 
default keyword arguments
-------------------------

+ "image_extn": ".png"

+ "matrix_color": "black"

+ "fiber_color": "white"

+ "matrix_edge_color": None

+ "fibre_edge_color" : None

+ "fibre_edge_thickness": None

+ "angle_units" : 'radians'

+ "z_comp_in_data" : False

+ "pixels" : (256, 256)

+ "get_imarray" : False

+ "img_type":"BINARY_SCALE"

+ "verbose" : 1

"""


def _set_default_plot_options(kwargs: dict) -> dict:
    for (default_kwarg, kw_val) in default_kwargs.items():
        if not default_kwarg in kwargs.keys():
            kwargs[default_kwarg] = kw_val


def plot_unit_cell(ruc_bbox,
                   inclusions_data,
                   image_path=None,
                   matrix_color: str = 'grey',
                   fibre_color: str = 'black',
                   matrix_edge_color: str = None,
                   fibre_edge_color: str = None,
                   fibre_edge_thickness=None,
                   angle_units: str = 'radians',
                   z_comp_in_data: bool = False,
                   pixels: tuple[int] = (100, 100),
                   get_imarray: bool = False,
                   img_type: str = "BINARY_SCALE",
                   verbose: int = 1
):
    """
    Plots RVE images of uni-directional composite.

    Arguments:
    ----------
    ruc_bbox: xmin, ymin, zmin, xmax, ymax, zmax
    inclusions_data: shape-data pairs Dict{str : ndarray},
    image_path: Optional[str] = None,
    h5_path: Optional[str] = None,
    matrix_color: str = 'grey',
    fibre_color: str = 'black',
    matrix_edge_color: str = None,
    fibre_edge_color: str = None,
    angle_units: str = 'radians',
    z_comp_in_data: bool = False,
    pixels: Tuple = (100, 100),
    my_dpi=96,

    Returns:
    --------
    Nothing

    """

    # _set_default_plot_options()  # FIXME make kwargs simpler


    fig = figure(0, figsize=(5, 5), frameon=False)
    ax = Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    #
    # plot RUC
    Plot2DShapes.plot_bbox(bounds=ruc_bbox,
                           fig_handle=gca(),
                           fc=matrix_color,
                           ec=matrix_edge_color, )
    # plot inclusions
    for (fibres_shape, inc_data) in inclusions_data.items():
        if fibres_shape.upper() == "CIRCLE":
            if z_comp_in_data:
                inc_data = inc_data[:, [0, 1, 3]]
            Plot2DShapes.plot_circular_discs(xyr=inc_data,
                                             fig_handle=gca(),
                                             ec=fibre_edge_color,
                                             fc=fibre_color,
                                             et=fibre_edge_thickness)
        elif fibres_shape.upper() == "CAPSULE":
            if z_comp_in_data:
                inc_data = inc_data[:, [0, 1, 3, 4, 5]]
            Plot2DShapes.plot_capsular_discs(xyt_ab=inc_data,
                                             fig_handle=gca(),
                                             ec=fibre_edge_color,
                                             fc=fibre_color,
                                             et=fibre_edge_thickness)
        elif fibres_shape.upper() == "ELLIPSE":
            if z_comp_in_data:
                inc_data = inc_data[:, [0, 1, 3, 4, 5]]
            Plot2DShapes.plot_elliptical_discs(xyt_ab=inc_data,
                                               fig_handle=gca(),
                                               ec=fibre_edge_color,
                                               fc=fibre_color,
                                               ang_units=angle_units,
                                               et=fibre_edge_thickness)
        elif fibres_shape.upper() == "RECTANGLE":
            if z_comp_in_data:
                inc_data = inc_data[:, [0, 1, 3, 4, 5, 6]]
            Plot2DShapes.plot_rectangles(xyt_abr=inc_data,
                                         fig_handle=gca(),
                                         ec=fibre_edge_color,
                                         fc=fibre_color,
                                         ang_units=angle_units,
                                         et=fibre_edge_thickness)
        elif fibres_shape.upper() == "REGULARPOLYGON":
            if z_comp_in_data:
                inc_data = inc_data[:, [0, 1, 3, 4, 5, 6]]
            Plot2DShapes.plot_regular_polygons(xyt_a_rf_n=inc_data,
                                               fig_handle=gca(),
                                               ec=fibre_edge_color,
                                               fc=fibre_color,
                                               ang_units=angle_units,
                                               et=fibre_edge_thickness)
        elif fibres_shape.upper() == "NLOBESHAPE":
            if z_comp_in_data:
                inc_data = inc_data[:, [0, 1, 3, 4, 5, 6]]
            Plot2DShapes.plot_nlobe_shapes(xyt_ro_rl_n=inc_data,
                                           fig_handle=gca(),
                                           ec=fibre_edge_color,
                                           fc=fibre_color,
                                           ang_units=angle_units,
                                           et=fibre_edge_thickness)
        elif fibres_shape.upper().startswith("NSTAR"):
            if z_comp_in_data:
                inc_data = inc_data[:, [0, 1, 3, 4, 5, 6, 7, 8]]
            Plot2DShapes.plot_stars(xyt_ro_rb_rtf_rbf_n=inc_data,
                                    fig_handle=gca(),
                                    ec=fibre_edge_color,
                                    fc=fibre_color,
                                    ang_units=angle_units,
                                    et=fibre_edge_thickness)
        elif fibres_shape.upper() == "CSHAPE":
            if z_comp_in_data:
                inc_data = inc_data[:, [0, 1, 3, 4, 5, 6, ]]
            Plot2DShapes.plot_cshapes(xyt_ro_ri_alpha=inc_data,
                                      fig_handle=gca(),
                                      ec=fibre_edge_color,
                                      fc=fibre_color,
                                      ang_units=angle_units,
                                      et=fibre_edge_thickness)

    # display or save
    axis("off")
    xlim([ruc_bbox[0], ruc_bbox[2]])
    ylim([ruc_bbox[1], ruc_bbox[3]])

    def get_imarray():
        io_buffer = io.BytesIO()
        savefig(io_buffer, format="raw")
        io_buffer.seek(0)
        imarray = reshape(frombuffer(io_buffer.getvalue(), dtype=uint8),
                          newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buffer.close()
        if img_type == "BINARY_SCALE":
            img_type_conv_id = "1"
        elif img_type == "GRAY_SCALE":
            img_type_conv_id = "L"
        elif img_type == None:
            img_type_conv_id = "RGB"
        return array(Image.fromarray(imarray
                                     ).convert(mode=img_type_conv_id
                                               ).resize(size=pixels, resample=Image.BICUBIC))
    #
    if image_path is None:
        return fig, get_imarray()
    else:
        image_path = str(image_path)
        imarray = get_imarray()
        Image.fromarray(imarray).save(image_path)
        clf()
        if verbose > 1:
            if verbose > 10:
                print(
                    f"image_array statistics:",
                    f"shape of the image array: {imarray.shape}",
                    f"Min pixel value: {amin(imarray)}",
                    f"Max pixel value: {amax(imarray)}",
                )

            print(f"image is saved in {img_type} mode at :: {image_path}")
        return


def plot_unit_cells_from_h5(
    h5file: h5py.File,
    images_dir: str,
    **kwargs,
):
    """_summary_

    :param h5file: _description_
    :type h5file: h5py.File
    :param images_dir: _description_
    :type images_dir: str
    """
    #
    _set_default_plot_options(kwargs)
    #
    with h5py.File(h5file, "r") as h5fid:
        for (dsID, ds) in h5fid.items():
            #
            data = {ak: transpose(av) for (ak, av) in ds.items()}
            #
            print(".", end="", flush=True)
            plot_unit_cell(
                ruc_bbox=(ds.attrs["xlb"], ds.attrs["ylb"],
                          ds.attrs["xub"], ds.attrs["yub"],),
                inclusions_data=data,
                image_path=path.join(
                    images_dir, f"{dsID}.{kwargs['image_extn']}"),
                matrix_color=kwargs["matrix_color"],
                fibre_color=kwargs["fibre_color"],
                matrix_edge_color=kwargs["matrix_edge_color"],
                fibre_edge_color=kwargs["fibre_edge_color"],
                fibre_edge_thickness=kwargs["fibre_edge_thickness"],
                angle_units='radians',
                z_comp_in_data=False,
                pixels=kwargs["pixels"],
                get_imarray=False,
                img_type=kwargs["img_type"],
                verbose=kwargs["verbose"],
            )


if __name__ == "__main__":
    print("Testing CShape")
    plot_unit_cell(
        ruc_bbox=(-10.0, -10.0, 10.0, 10.0),
        inclusions_data={"CSHAPE": [
            [1.0, 1.0, 0.0 * pi, 6.0, 3.0, 0.5*pi, ]], },
        image_path=path.join(path.expanduser("~"), "test_cshape.png"),
        fibre_color="0.1",
        matrix_color="0.6",
    )
