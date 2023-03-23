import bmcs_utils.api as bu
import numpy as np

from bmcs_shell.folding.geometry.wb_tessellation.wb_num_tessellation_base import WBNumTessellationBase
from bmcs_shell.folding.geometry.wb_tessellation.wb_num_tessellation_grad_base import WBNumTessellationGradBase
import traits.api as tr

class WBNumTessellationGradInvest(WBNumTessellationGradBase):
    """ A class to investigate the angles for tessellating three cells manually. """

    name = 'WBNumTessellationInvest'

    rot_br = bu.Float(0.5)
    rot_ur = bu.Float(0.5)
    rot_u = bu.Float(0.5)
    investigate_rot = bu.Bool
    wb_cell_1_idx = bu.Int(0, GEO=True)
    wb_cell_2_idx = bu.Int(1, GEO=True)

    ipw_view = bu.View(
        *WBNumTessellationGradBase.ipw_view.content,
        bu.Item('investigate_rot'),
        bu.Item('rot_br', latex=r'rot~br', editor=bu.FloatRangeEditor(low=0, high=2 * np.pi, n_steps=150,
                                                                      continuous_update=True)),
        bu.Item('rot_ur', latex=r'rot~ur', editor=bu.FloatRangeEditor(low=0, high=2 * np.pi, n_steps=150,
                                                                      continuous_update=True)),
        bu.Item('rot_u', latex=r'rot~u', editor=bu.FloatRangeEditor(low=0, high=2 * np.pi, n_steps=150,
                                                                      continuous_update=True)),
        bu.Item('wb_cell_1_idx'),
        bu.Item('wb_cell_2_idx'),
    )

    def setup_plot(self, pb):
        super().setup_plot(pb)

    def update_plot(self, pb):
        if self.k3d_mesh:
            cell_1 = self.wb_cells[self.wb_cell_1_idx]
            cell_2 = self.wb_cells[self.wb_cell_2_idx]
            X_Ia = cell_1.X_Ia.astype(np.float32)

            print('br_X_Ia')
            br_X_Ia = self._get_br_X_Ia(cell_1.X_Ia, self.rot_br if self.investigate_rot else None,
                                        X2_Ia=cell_2.X_Ia)
            print('ur_X_Ia')
            ur_X_Ia = self._get_ur_X_Ia(cell_1.X_Ia, self.rot_ur if self.investigate_rot else None,
                                        X2_Ia=cell_2.X_Ia)
            print('u_X_Ia')
            u_X_Ia = self._get_ul_X_Ia(ur_X_Ia, rot=self.rot_u if self.investigate_rot else None,
                                       X2_Ia=cell_1.X_Ia)

            # TODO: make this cleaner, this is just to have all data in X_Ia_shell and I_Fi_shell for obj export
            n = 4
            mesh_X_nIa = np.zeros((n * 7, 3)).astype(np.float32)
            mesh_X_nIa[0:7, :] = X_Ia
            mesh_X_nIa[7:14, :] = br_X_Ia
            mesh_X_nIa[14:21, :] = ur_X_Ia
            mesh_X_nIa[21:, :] = u_X_Ia

            # Calculating mesh_I_Fi
            I_Fi = self.wb_cells[0].I_Fi
            seven_mult_i = 7 * np.arange(n)
            seven_mult_Iab = seven_mult_i[:, np.newaxis, np.newaxis] + np.zeros((n, 6, 3), dtype=np.int32)
            mesh_I_nFi = np.full((n, 6, 3), I_Fi)
            mesh_I_nFi = mesh_I_nFi + seven_mult_Iab
            mesh_I_nFi = mesh_I_nFi.reshape((n * 6, 3))

            self.X_Ia_shell = mesh_X_nIa.astype(np.float32)
            self.I_Fi_shell = mesh_I_nFi.astype(np.uint32)

            # This is a hack, k3d_mesh['X_Ia'] was meant to be the
            self.k3d_mesh['X_Ia'].vertices = self.X_Ia_shell
            self.k3d_mesh['X_Ia'].indices = self.I_Fi_shell
            self.k3d_wireframe['X_Ia'].vertices = self.X_Ia_shell
            self.k3d_wireframe['X_Ia'].indices = self.I_Fi_shell

            self.k3d_mesh['br_X_Ia'].vertices = []
            self.k3d_mesh['ur_X_Ia'].vertices = []
            self.k3d_wireframe['br_X_Ia'].vertices = []
            self.k3d_wireframe['ur_X_Ia'].vertices = []
        else:
            self.setup_plot(pb)
