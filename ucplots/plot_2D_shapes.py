"""
Module for plotting Unit Cells defined by matrix and inclusion phases, for
example RVEs of composite materials.

plot_uc() is the main function for plotting the unit cells


"""

import numpy as np
from numpy import array, column_stack, concatenate, append, linspace
from numpy import rad2deg, deg2rad, sin, cos, tan, arcsin
from numpy import pi, sqrt, sign
from matplotlib.pyplot import figure, gca, savefig, show


# ==========================================================================
#                               Shape definitions
# ==========================================================================


class Shape2D:
    """
    It creates the standard 2D shapes and generated points sufficient for plotting shapes
    """

    def __init__(self,
                 centre=(0.0, 0.0),
                 ref_angle=0.0,
                 angle_units: str = "radians"):
        """

        :param Tuple centre: x,y coordinates of the shape  centre
        :param str angle_units: Radians

        """
        self.xc = centre[0]  # x-coordinates of shape centre
        self.yc = centre[1]  # y-coordinates of shape centre
        self.xy = array([])  # xy - coordinates of the shape
        if angle_units.lower() == "radians":
            self.theta = ref_angle
        elif angle_units.lower() == "degrees":
            self.theta = deg2rad(ref_angle)  # defaults to radians
        else:
            raise ValueError(
                "Angle units must be either 'radians' or 'degrees'")
        #
        self.ang_units = "radians"
        return

    def __repr__(self):
        return "2D_shape_patch"

    def __str__(self):
        return "2D_shape_patch"

    #
    @staticmethod
    def _set_ang_units(ang, _from="degrees", _to="radians"):
        err_msg = "Angle units must be either 'radians' or 'degrees' but given as {},{}".format(
            _from, _to)
        if _from.lower() != _to.lower():
            if _from.lower() == "degrees":
                if _to.lower() == "radians":
                    return deg2rad(ang)
                else:
                    raise ValueError(err_msg)
            elif _from.lower() == "radians":
                if _to.lower() == "degrees":
                    return rad2deg(ang)
                else:
                    raise ValueError(err_msg)
            else:
                raise ValueError(err_msg)
        else:
            return ang

    def _rot_mat(self, angle=None):
        """Private method to return rotation matrix
        :return: rotation matrix
        """
        #
        if angle is None:
            angle = self.theta
        return [[+cos(angle), sin(angle)],
                [-sin(angle), cos(angle)], ]

    def _trans_mat(self):
        """ Private method for returning translation matrix

        :return:
        """
        return [self.xc, self.yc]

    def _rot_trans_mat(self, x_y, angle=None):
        return (x_y @ self._rot_mat(angle=angle)) + self._trans_mat()

    def make_rectangle(self,
                       semi_mjl=2.0,
                       semi_mnl=1.0,
                       cr=0.0,
                       num_sect_points=100,
                       ang_units="radians", ):
        """ It creates a regular polygon 2D shape,

        :param ang_units:
        :param float semi_mjl:
        :param float semi_mnl:
        :param float cr: corner radius
        :param int num_sect_points:
        :return: Shape2D

        """
        # Making the rectangle at origin, i.e., reference rectangle
        self.xy = array([]).reshape(0, 2)
        alpha = 0.25 * pi
        for i in range(4):
            # c_theta : corner theta  # FIXME
            c_theta = (((2.0 * i) + 1.0) * alpha)
            theta_sector = linspace(
                start=c_theta - alpha, stop=c_theta + alpha, num=num_sect_points)
            xx_yy = [(semi_mjl - cr) * sign(cos(c_theta)), (semi_mnl - cr) * sign(sin(c_theta))
                     ] + (cr * column_stack([cos(theta_sector), sin(theta_sector)]))
            self.xy = concatenate((self.xy, xx_yy), axis=0)
        # Transforming to the desired state.
        self.xy = self._rot_trans_mat(self.xy)
        return self

    def make_regular_polygon(self,
                             side_len,
                             r_corner,
                             num_sides,
                             num_sect_points=100,
                             ang_units="radians", ):
        """

        :param side_len:
        :param r_corner:
        :param num_sides:
        :param alpha_0:
        :param num_sect_points:
        :param ang_units:
        :return:
        """
        #
        theta = self._set_ang_units(self.theta, ang_units)
        self.xy = array([]).reshape(0, 2)
        alpha = pi / num_sides
        #
        r_inscribed = 0.5 * side_len / tan(alpha)
        h = (r_inscribed - r_corner) / cos(alpha)
        # creating the reference polygon at origin
        for i in range(int(num_sides)):
            theta_vertex = theta + (2.0 * i * alpha)
            theta_sector = linspace(
                start=theta_vertex - alpha, stop=theta_vertex + alpha, num=num_sect_points)
            xx_yy = [self.xc + (h * cos(theta_vertex)),
                     self.yc + (h * sin(theta_vertex))] + (r_corner * column_stack([cos(theta_sector),
                                                                                    sin(theta_sector)]))
            self.xy = concatenate((self.xy, xx_yy), axis=0)
        #
        # returning rotated and translated reference-regular polygon
        # self.xy = self._rot_trans_mat(x_y)
        return self

    #
    def make_elliptical_sector(self,
                               semi_mjl=2.0,
                               semi_mnl=1.0,
                               theta_1=0.0,
                               theta_2=pi,
                               ang_units='radians',
                               num_sec_points=100,
                               direct_loop_closure=False):
        """

        :param direct_loop_closure:
        :param num_sec_points:
        :param semi_mnl:
        :param semi_mjl:
        :param ang_units:
        :param float theta_1: starting angle of the sector
        :param float theta_2: ending angle of the sector
        :return: Shape2D
        :rtype Shape2D:
        """
        assert semi_mjl >= semi_mnl, "Ensure that major axis length >= minor axis length."
        theta_1 = self._set_ang_units(theta_1, ang_units, _to="radians")
        theta_2 = self._set_ang_units(theta_2, ang_units, _to="radians")
        #
        theta_i = linspace(start=theta_1, stop=theta_2, num=num_sec_points)
        x_y = column_stack([semi_mjl * cos(theta_i), semi_mnl * sin(theta_i)])
        if direct_loop_closure:
            x_y = append(x_y, [x_y[0]], axis=0)
        else:
            x_y = append(x_y, [[0.0, 0.0], x_y[0]], axis=0)
        #
        # returning rotated and translated reference-elliptical sector
        self.xy = self._rot_trans_mat(x_y)
        return self

    #
    def make_cshape(
            self,
            out_radius=2.0,
            in_radius=1.0,
            alpha=pi,
            num_sect_points=100):

        tip_radius = 0.5 * (out_radius - in_radius)
        mean_radius = 0.5 * (out_radius + in_radius)

        first_tip_centre = [
            self.xc + mean_radius * cos(self.theta),
            self.yc + mean_radius * sin(self.theta)
        ]
        last_tip_centre = [
            self.xc + mean_radius * cos(self.theta + alpha),
            self.yc + mean_radius * sin(self.theta + alpha)
        ]

        outer_sector_theta = linspace(start=self.theta, stop=(
                self.theta + alpha), num=num_sect_points)
        last_tip_sector_theta = linspace(
            start=(self.theta + alpha), stop=(self.theta + alpha + pi), num=num_sect_points)
        inner_sector_theta = linspace(
            start=(self.theta + alpha), stop=self.theta, num=num_sect_points)
        first_tip_sector_theta = linspace(
            start=(self.theta + pi), stop=(self.theta + (2.0 * pi)), num=num_sect_points)

        outer_sector_xx_yy = [self.xc, self.yc] + (out_radius * column_stack(
            [cos(outer_sector_theta), sin(outer_sector_theta)]))
        last_tip_xx_yy = last_tip_centre + \
                         (tip_radius *
                          column_stack([cos(last_tip_sector_theta), sin(last_tip_sector_theta)]))
        inner_sector_xx_yy = [self.xc, self.yc] + (in_radius * column_stack(
            [cos(inner_sector_theta), sin(inner_sector_theta)]))
        first_tip_xx_yy = first_tip_centre + \
                          (tip_radius *
                           column_stack([cos(first_tip_sector_theta), sin(first_tip_sector_theta)]))

        self.xy = concatenate(
            (outer_sector_xx_yy, last_tip_xx_yy, inner_sector_xx_yy, first_tip_xx_yy), axis=0)
        return self

    #
    def make_capsule(self,
                     semi_mjl=2.0,
                     semi_mnl=1.0,
                     num_sect_points=100):
        """Generate the coordinates for describing a capsule shape

        :param num_sect_points:
        :param float semi_mjl:
        :param float semi_mnl:
        :return:
        """
        alpha = pi / 2.0
        theta_sector_1 = linspace(
            start=1.0 * alpha, stop=3.0 * alpha, num=num_sect_points)
        xx_yy_1 = [-(semi_mjl - semi_mnl), 0.0] + (semi_mnl *
                                                   column_stack([cos(theta_sector_1), sin(theta_sector_1)]))
        theta_sector_2 = linspace(
            start=3.0 * alpha, stop=5.0 * alpha, num=num_sect_points)
        xx_yy_2 = [+(semi_mjl - semi_mnl), 0.0] + (semi_mnl *
                                                   column_stack([cos(theta_sector_2), sin(theta_sector_2)]))
        self.xy = self._rot_trans_mat(concatenate((xx_yy_1, xx_yy_2), axis=0))
        return self
        #

    def make_circle(self,
                    radius=1.0):
        """Generate the coordinates for describing a circle shape

        :param radius:
        :return:
        """

        return self.make_elliptical_sector(semi_mjl=radius, semi_mnl=radius,
                                           theta_1=0.0, theta_2=2.0 * pi,
                                           ang_units="radians",
                                           direct_loop_closure=True)

    #
    def make_ellipse(self,
                     semi_mjl=2.0,
                     semi_mnl=1.0, ):
        """Generate the coordinates for describing a ellipse shape

        :param semi_mjl:
        :param semi_mnl:
        :return:
        """
        return self.make_elliptical_sector(semi_mjl, semi_mnl,
                                           theta_1=0.0,
                                           theta_2=2.0 * pi,
                                           ang_units="radians",
                                           direct_loop_closure=True)

    #

    def make_lobe(self, num_lobes, ro, rl, sector_resolution=100, ):
        """
        For nlobe shape with its centre at origin, for each lobe,
            Sector 1: centre (ro - rl, 0.0), radius rl, angle 2(alpha + theta) in CCW direction
            Sector 2: centre (b*cos(alpha), b*sin(alpha)), radius rl, angle 2(alpha + theta) in CCW direction
        """
        num_lobes = int(num_lobes)
        alpha = pi / num_lobes
        theta = arcsin(0.5 * (ro - rl) * sin(alpha) / rl)
        b = 2.0 * rl * sin(alpha + theta) / sin(alpha)
        #
        sector1_theta = linspace(
            start=(- alpha - theta), stop=(alpha + theta), num=sector_resolution)
        sector2_theta = linspace(
            start=(pi + alpha + theta), stop=(pi + alpha - theta), num=sector_resolution)
        #
        xy_1stlobe_sector1 = column_stack(
            [(ro - rl) + (rl * cos(sector1_theta)), (rl * sin(sector1_theta))]
        )
        xy_1stlobe_sector2 = column_stack(
            [b * cos(alpha) + (rl * cos(sector2_theta)), b *
             sin(alpha) + (rl * sin(sector2_theta))]
        )
        xy_1stlobe = concatenate(
            (xy_1stlobe_sector1, xy_1stlobe_sector2,), axis=0)
        #
        x_y = array([]).reshape(0, 2)
        for i in range(num_lobes):
            theta_lobe = 2.0 * i * alpha
            xx_yy = xy_1stlobe @ self._rot_mat(angle=theta_lobe)
            x_y = concatenate((x_y, xx_yy), axis=0)
        #
        # returning rotated and translated reference-elliptical sector
        self.xy = self._rot_trans_mat(x_y)
        return self

    #

    def make_star(self, num_tips, ro, rb, tip_fr, base_fr,
                  sector_resolution=100, ):
        """

        :param base_fr: base fillet radius
        :param sector_resolution:
        :param int num_tips:
        :param float ro: Outer radius
        :param float rb: base radius
        :param float tip_fr: tip fillet radius
        :return:
        """
        num_tips = int(num_tips)
        alpha = pi / num_tips
        x0 = x1 = rb * cos(alpha)
        y0 = rb * sin(-alpha)
        y1 = rb * sin(alpha)
        # beta evaluation
        a = rb + base_fr
        b = ro - tip_fr
        c = base_fr + tip_fr
        d = (a * sin(alpha))
        e = (a * cos(alpha)) - b
        beta = arcsin(
            ((e * c) + (d * sqrt((d * d) + (e * e) - (c * c)))) / ((e * e) + (d * d)))
        #
        theta_b1 = linspace(start=pi - alpha, stop=(0.5 *
                                                    pi) + beta, num=sector_resolution)
        first_sector_theta = linspace(
            start=-(pi / 2) + beta, stop=(pi / 2) - beta, num=sector_resolution)
        theta_b2 = linspace(start=(1.5 * pi) - beta,
                            stop=pi + alpha, num=sector_resolution)

        #
        xy_ft_1 = column_stack([((rb + base_fr) * cos(alpha)) + (base_fr * cos(theta_b1)),
                                -((rb + base_fr) * sin(alpha)) + (base_fr * sin(theta_b1))])
        xy_ft_s = column_stack([(ro - tip_fr) + (tip_fr * cos(first_sector_theta)),
                                tip_fr * sin(first_sector_theta)])
        xy_ft_2 = column_stack([((rb + base_fr) * cos(alpha)) + (base_fr * cos(theta_b2)),
                                ((rb + base_fr) * sin(alpha)) + (base_fr * sin(theta_b2))])
        #
        xy_first_tip = concatenate((xy_ft_1, xy_ft_s, xy_ft_2), axis=0)
        #
        x_y = array([]).reshape(0, 2)
        for i in range(num_tips):
            theta_tip = i * 2.0 * alpha
            xx_yy = xy_first_tip @ self._rot_mat(angle=theta_tip)
            x_y = concatenate((x_y, xx_yy), axis=0)
        #
        # returning rotated and translated reference-elliptical sector
        self.xy = self._rot_trans_mat(x_y)
        return self

    #
    def make_bbox(self, bounds=(-1.0, -1.0, 1.0, 1.0)):
        """Generate the coordinates for describing a bbox

        :param Tuple bounds:
        :return: Shape2D
        """
        #
        xlb, ylb, xub, yub = bounds
        self.xy = array([[xlb, xub, xub, xlb, xlb], [
            ylb, ylb, yub, yub, ylb]]).T
        return self
        #


# ==========================================================================
#                               Plotting
# ==========================================================================


class Plot2DShapes(Shape2D):
    """

    """

    def __init__(self,
                 centre=(0.0, 0.0),
                 ref_angle=0.0,
                 angle_units: str = "radians",
                 ec='k',
                 fc='grey',
                 et=1.0, ):
        """

        :param Tuple centre: x,y coordinates of the shape  centre
        :param str or None ec:
        :param str or None fc:
        :param str angle_units: Radians

        """
        self.fc = fc
        self.ec = ec
        self.et = et
        super().__init__(centre, ref_angle, angle_units)
        #
        return

    #
    def plot(self, fig_handle=None, ):
        """

        :param fig_handle:
        :return: fig_handle
        """
        #
        if fig_handle is None:
            figure(0)
            fig_handle = gca()
        #
        # making the plot
        fig_handle.fill(self.xy[:, 0], self.xy[:, 1],
                        facecolor=self.fc,
                        edgecolor=self.ec,
                        linewidth=self.et,
                        antialiased=True,
                        )
        return fig_handle

    @staticmethod
    def save(fig_path=None, ):
        savefig(fig_path)
        return

    @staticmethod
    def show():
        show()
        return

    """
                        CLASS METHODS FOR PLOTTING MANY SHAPES 
    """

    @classmethod
    def plot_rectangles(cls, fig_handle, xyt_abr,
                        ec='k',
                        fc='grey',
                        et=1.0,
                        ang_units='radians'):
        #
        xyt_abr[:, 2:3] = Shape2D._set_ang_units(
            xyt_abr[:, 2:3], _from=ang_units, _to="radians")
        for (axc, ayc, ath, aa, ab, ar) in xyt_abr:
            patch = cls(centre=(axc, ayc), ref_angle=ath,
                        angle_units=ang_units, ec=ec, fc=fc, et=et)
            a_rect = patch.make_rectangle(semi_mjl=aa, semi_mnl=ab, cr=ar)
            fig_handle = a_rect.plot(fig_handle=fig_handle)
        #
        return fig_handle

    @classmethod
    def plot_regular_polygons(cls, fig_handle, xyt_a_rf_n,
                              ec='k',
                              fc='grey',
                              et=1.0,
                              ang_units='radians'):
        #
        xyt_a_rf_n[:, 2:3] = Shape2D._set_ang_units(
            xyt_a_rf_n[:, 2:3], _from=ang_units, _to="radians")
        for (axc, ayc, ath, aa, ar, an) in xyt_a_rf_n:
            patch = cls(centre=(axc, ayc), ref_angle=ath,
                        angle_units=ang_units, ec=ec, fc=fc, et=et)
            a_reg_polygon = patch.make_regular_polygon(
                num_sides=an, side_len=aa, r_corner=ar)
            fig_handle = a_reg_polygon.plot(fig_handle=fig_handle)
        #
        return fig_handle

    @classmethod
    def plot_elliptical_discs(cls, fig_handle, xyt_ab,
                              ec='k',
                              fc=None,
                              et=1.0,
                              ang_units: str = 'radians'):
        """

        """
        #
        xyt_ab[:, 2:3] = Shape2D._set_ang_units(
            xyt_ab[:, 2:3], _from=ang_units, _to="radians")
        #
        for (axc, ayc, ath, aa, ab) in xyt_ab:
            patch = cls(centre=(axc, ayc), ref_angle=ath,
                        angle_units=ang_units, ec=ec, fc=fc, et=et)
            an_ellipse = patch.make_ellipse(semi_mjl=aa, semi_mnl=ab, )
            fig_handle = an_ellipse.plot(fig_handle=fig_handle)
        #
        return fig_handle

    @classmethod
    def plot_circular_discs(cls, fig_handle, xyr,
                            ec=None,
                            et=1.0,
                            fc='grey', ):
        """
        """
        for (axc, ayc, ar) in xyr:
            patch = cls(centre=(axc, ayc), ec=ec, fc=fc, et=et)
            a_circle = patch.make_circle(radius=ar)
            fig_handle = a_circle.plot(fig_handle=fig_handle)
        return fig_handle

    @classmethod
    def plot_capsular_discs(cls, fig_handle, xyt_ab,
                            ec='k',
                            fc=None,
                            et=1.0,
                            ang_units: str = "radians"):
        """

        :param fig_handle:
        :param xyt_ab:
        :param ec:
        :param fc:
        :param ang_units:
        :return:

        Parameters
        ----------
        et
        """
        for (axc, ayc, ath, aa, ab) in xyt_ab:
            patch = cls(centre=(axc, ayc), ref_angle=ath,
                        angle_units=ang_units, ec=ec, fc=fc, et=et)
            a_capsule = patch.make_capsule(semi_mjl=aa, semi_mnl=ab)
            fig_handle = a_capsule.plot(fig_handle=fig_handle)
        return fig_handle

    @classmethod
    def plot_stars(cls, fig_handle, xyt_ro_rb_rtf_rbf_n,
                   ec='k',
                   fc=None,
                   et=1.0,
                   ang_units: str = "radians"):
        for (axc, ayc, ath, aro, arb, a_rtf, a_rbf, ant) in xyt_ro_rb_rtf_rbf_n:
            patch = cls(centre=(axc, ayc), ref_angle=ath,
                        angle_units=ang_units, ec=ec, fc=fc, et=et)
            a_star = patch.make_star(
                num_tips=ant, ro=aro, rb=arb, tip_fr=a_rtf, base_fr=a_rbf)
            fig_handle = a_star.plot(fig_handle=fig_handle)
        return fig_handle

    @classmethod
    def plot_nlobe_shapes(cls, fig_handle, xyt_ro_rl_n,
                          ec='k',
                          fc=None,
                          et=1.0,
                          ang_units: str = "radians"):
        for (axc, ayc, ath, aro, arl, anl) in xyt_ro_rl_n:
            patch = cls(centre=(axc, ayc), ref_angle=ath,
                        angle_units=ang_units, ec=ec, fc=fc, et=et)
            a_nlobe = patch.make_lobe(num_lobes=anl, ro=aro, rl=arl)
            fig_handle = a_nlobe.plot(fig_handle=fig_handle)
        return fig_handle

    @classmethod
    def plot_cshapes(cls, fig_handle, xyt_ro_ri_alpha,
                     ec='k',
                     fc=None,
                     et=1.0,
                     ang_units: str = "radians"):
        for (axc, ayc, ath, aro, ari, a_alpha) in xyt_ro_ri_alpha:
            patch = cls(centre=(axc, ayc), ref_angle=ath,
                        angle_units=ang_units, ec=ec, fc=fc, et=et)
            a_csh = patch.make_cshape(
                out_radius=aro, in_radius=ari, alpha=a_alpha)
            fig_handle = a_csh.plot(fig_handle=fig_handle)
        return fig_handle

    @classmethod
    def plot_bbox(cls, fig_handle, bounds, ec='k', fc='grey'):
        """

        :param fc:
        :param ec:
        :param fig_handle:
        :param bounds:
        :return:
        """
        cls(ec=ec, fc=fc).make_bbox(bounds).plot(fig_handle)
        return fig_handle

# ==========================================================================
#                               Main Function
# ==========================================================================
