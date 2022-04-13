import bmcs_utils.api as bu
import numpy as np

from bmcs_shell.folding.geometry.wb_tessellation.wb_num_tessellation_base import WBNumTessellationBase


class WBNumTessellationInvest(WBNumTessellationBase):
    """ A class to investigate the angles for tessellating three cells manually. """

    name = 'WBNumTessellationInvest'

    rot_br = bu.Float(0.5)
    rot_ur = bu.Float(0.5)
    investigate_rot = bu.Bool

    ipw_view = bu.View(
        *WBNumTessellationBase.ipw_view.content,
        bu.Item('investigate_rot'),
        bu.Item('rot_br', latex=r'rot~br', editor=bu.FloatRangeEditor(low=0, high=2 * np.pi, n_steps=150,
                                                                      continuous_update=True)),
        bu.Item('rot_ur', latex=r'rot~ur', editor=bu.FloatRangeEditor(low=0, high=2 * np.pi, n_steps=150,
                                                                      continuous_update=True)),
    )

    def setup_plot(self, pb):
        super().setup_plot(pb)

    def update_plot(self, pb):
        if self.k3d_mesh:
            sol = self.sol
            X_Ia = self.X_Ia.astype(np.float32)
            br_X_Ia = self._get_br_X_Ia(self.X_Ia, self.rot_br if self.investigate_rot else sol[
                0]).astype(np.float32)
            ur_X_Ia = self._get_ur_X_Ia(self.X_Ia, self.rot_ur if self.investigate_rot else sol[
                1]).astype(np.float32)
            self.k3d_mesh['X_Ia'].vertices = X_Ia
            self.k3d_mesh['br_X_Ia'].vertices = br_X_Ia
            self.k3d_mesh['ur_X_Ia'].vertices = ur_X_Ia
            self.k3d_wireframe['X_Ia'].vertices = X_Ia
            self.k3d_wireframe['br_X_Ia'].vertices = br_X_Ia
            self.k3d_wireframe['ur_X_Ia'].vertices = ur_X_Ia
        else:
            self.setup_plot(pb)
