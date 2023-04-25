"""Controlled-source electromagnetic (CSEM) survey (patch collection) data."""
import os.path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
from saem import Mare2dEMData
from saem import CSEMData


class CSEMSurvey():
    """Class for (multi-patch/transmitter) CSEM data."""

    def __init__(self, arg=None, **kwargs):
        """Initialize class with either data filename or patch instance.

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.
        """
        self.patches = []
        self.origin = [0, 0, 0]
        self.angle = 0.
        self.basename = "new"
        if arg is not None:
            if isinstance(arg, str):
                if arg.endswith(".emdata"):  # obviously a Mare2d File
                    self.importMareData(arg, **kwargs)
                elif arg.endswith(".npz"):
                    self.loadNPZ(arg, **kwargs)
            elif hasattr(arg, "__iter__"):
                for a in arg:
                    self.addPatch(a)
            elif isinstance(arg, CSEMData):
                self.addPatch(arg)
            else:
                raise TypeError("Cannot use type")

    def __repr__(self):
        """String representation."""
        st = "CSEMSurvey class with {:d} patches".format(len(self.patches))
        for i, p in enumerate(self.patches):
            st = "\n".join([st, p.__repr__()])

        return st

    def __getitem__(self, i):
        """Return property (str) or patch number (int)."""
        if isinstance(i, int):
            return self.patches[i]
        elif isinstance(i, str):
            return getattr(self, i)

    def loadNPZ(self, filename, mtdata=False, **kwargs):
        """Load numpy-compressed (NPZ) file."""
        ALL = np.load(filename, allow_pickle=True)
        self.f = ALL["freqs"]

        a = 0

        if hasattr(ALL, "line") and hasattr(ALL["line"], "len"):
            line = ALL["line"]  # NpzFile??
        else:
            line = np.array([], dtype=int)
            for i in range(len(ALL["DATA"])):
                line = np.append(line, np.ones(len(ALL["DATA"][i]['rx']),
                                               dtype=int))

        for i in range(len(ALL["DATA"])):
            if not mtdata:
                patch = CSEMData()
            else:
                pass
                # patch = MTData()

            patch.extractData(ALL, i)
            self.addPatch(patch)
            if hasattr(line, 'len') and len(line) > 0:
                patch.line = line[a:a+len(patch.rx)]
                a += len(patch.rx)
            else:
                patch.line = np.zeros_like(patch.rx)
                patch.detectLines()

    def importMareData(self, mare, flipxy=False, **kwargs):
        """Import Mare2dEM file."""
        if isinstance(mare, str):
            self.basename = mare.replace(".emdata", "")
            mare = Mare2dEMData(mare, flipxy=flipxy)

        tI = kwargs.setdefault('tI', np.arange(len(mare.txPositions())))
        for typ in ["E", "B"]:
            for i in tI:
                part = mare.getPart(tx=i+1, typ=typ, clean=True)
                if len(part.DATA) == 0:  # no data
                    break

                txl = mare.txpos[i, 3]
                # azimuth and dip need to be used as well !!!
                txpos = np.array([[mare.txpos[i, 0], mare.txpos[i, 0]],
                                  mare.txpos[i, 1] + np.array([-1/2, 1/2])*txl,
                                  [0, 0]])

                if "txs" in kwargs:
                    txpos = kwargs["txs"][i].T

                fak = 1e9 if typ == "B" else 1
                mats = [part.getDataMatrix(field=typ+"x") * txl * fak,
                        part.getDataMatrix(field=typ+"y") * txl * fak,
                        -part.getDataMatrix(field=typ+"z") * txl * fak]
                errs = [part.getDataMatrix(field=typ+"x",
                                           column="StdErr") * txl * fak,
                        part.getDataMatrix(field=typ+"y",
                                           column="StdErr") * txl * fak,
                        part.getDataMatrix(field=typ+"z",
                                           column="StdErr") * txl * fak]

                rx, ry, rz = part.rxpos.T
                cs = CSEMData(f=np.array(mare.f), rx=rx, ry=ry, rz=rz,
                              txPos=txpos, mode=typ)
                cs.basename = "patch{:d}".format(i+1)
                for i, mat in enumerate(mats):
                    if mat.shape[0] == 0:
                        cs.cmp[i] = 0
                        mats[i] = np.zeros((len(part.f), part.rxpos.shape[0]))
                        errs[i] = np.zeros_like(mats[i])

                cs.DATA = np.stack(mats)
                cs.ERR = np.stack(errs)
                cs.chooseActive()
                self.addPatch(cs)

    def addPatch(self, patch, name=None):
        """Add a new patch to the file.

        Parameters
        ----------
        patch : CSEMData | str
            CSEMData instance or string to load into that
        """
        if isinstance(patch, str):
            patch = CSEMData(patch)
            if name is not None:
                patch.basename = name

        if len(self.patches) == 0:
            self.angle = patch.angle
            self.origin = patch.origin
            self.cmp = patch.cmp
            print('  -  copy *angle*, *origin* and *cmp* from first patch  -')
        else:
            assert self.angle == patch.angle, "angle not matching"
            assert np.allclose(self.origin, patch.origin), "origin not equal"
            if hasattr(self, 'f'):
                assert len(self.f) == len(patch.f), "frequency number unequal"
                f1 = np.round(self.f, 1)
                f2 = np.round(patch.f, 1)
                assert np.allclose(f1, f2), \
                    "frequencies not equal" + f1.__str__()+" vs. "+f2.__str__()

        self.patches.append(patch)

    def add(self, *args, **kwargs):
        """Alias for addPatch."""  # maybe do it the other way round
        self.addPatch(*args, **kwargs)

    def showPositions(self, **kwargs):
        """Show all positions."""
        fig, ax = plt.subplots()
        ma = ["x", "+", "^", "v"]
        for i, p in enumerate(self.patches):
            p.showPos(ax=ax, color="C{:d}".format(i),
                      marker=ma[i % len(ma)], **kwargs)

        return fig, ax

    def showData(self, **kwargs):
        """."""
        for i, p in enumerate(self.patches):
            p.showData(**kwargs)

    def setOrigin(self, *args, **kwargs):
        """Set the same origin for all patches (reshifting if existing)."""
        for p in self.patches:
            p.setOrigin(*args, **kwargs)

    def filter(self, *args, **kwargs):
        """Filter."""
        for p in self.patches:
            p.filter(*args, **kwargs)

    def estimateError(self, *args, **kwargs):
        """Estimate error model."""
        for p in self.patches:
            p.estimateError(*args, **kwargs)

    def getData(self, line=None, **kwargs):
        """Gather data from individual patches."""
        DATA = []
        lines = np.array([])
        for i, p in enumerate(self.patches):
            data = p.getData(line=line, **kwargs)
            data["tx_ids"] = [i]  # for now fixed, later global list
            DATA.append(data)
            lines = np.concatenate((lines, p.line+(i+1)*100))

        return DATA, lines

    def saveData(self, fname=None, line=None, **kwargs):
        """Save data in numpy format for 2D/3D inversion."""
        cmp = kwargs.setdefault("cmp", self.patches[0].cmp)
        if fname is None:
            fname = self.basename
            if line is not None:
                fname += "-line" + str(line)

            for ci, cid in enumerate(cmp):
                if cid:
                    fname += self.patches[0].cstr[ci]
        else:
            if fname.startswith("+"):
                fname = self.basename + "-" + fname

        txs = [np.column_stack((p.tx, p.ty, p.ty*0)) for p in self.patches]
        DATA, lines = self.getData(line=line, **kwargs)
        np.savez(fname+".npz",
                 tx=txs,
                 freqs=self.patches[0].f,
                 DATA=DATA,
                 line=lines,
                 cmp=[patch["cmp"] for patch in DATA],
                 origin=self.origin,  # global coordinates with altitude
                 rotation=self.angle)

    def loadResponse(self, dirname=None, response=None):
        """Load model response file."""
        if not dirname.endswith('/'):
            dirname += '/'

        if response is None:
            respfiles = sorted(glob(dirname+"response_iter*.npy"))
            if len(respfiles) == 0:
                pg.error("Could not find response file")

            responseVec = np.load(respfiles[-1])
            respR, respI = np.split(responseVec, 2)
            response = respR + respI*1j

        ind = np.hstack((0, np.cumsum([p.nData() for p in self.patches])))
        for i, p in enumerate(self.patches):
            p.loadResponse(response=response[ind[i]:ind[i+1]])

    def loadResults(self, dirname=None, datafile=None, invmesh="Prisms",
                    jacobian=None):
        """Load inversion results from directory."""
        datafile = datafile or self.basename
        if dirname is None:
            dirname = datafile + "_" + invmesh + "/"
        if dirname[-1] != "/":
            dirname += "/"

        if os.path.exists(dirname + "inv_model.npy"):
            self.model = np.load(dirname + "inv_model.npy")
        else:
            self.model = np.load(sorted(glob(dirname+"sig_iter_*.npy"))[0])

        self.chi2s = np.loadtxt(dirname + "chi2.dat", usecols=3)
        self.loadResponse(dirname)
        self.J = None
        if os.path.exists(dirname+"invmesh.vtk"):
            self.mesh = pg.load(dirname+"invmesh.vtk")
        else:
            self.mesh = pg.load(dirname + datafile + "_final_invmodel.vtk")

        jacobian = jacobian or datafile+"_jacobian.bmat"
        jname = dirname + jacobian
        if os.path.exists(jname):
            self.J = pg.load(jname)
            print("Loaded jacobian: "+jname, self.J.rows(), self.J.cols())
        elif os.path.exists(dirname+"jacobian.bmat"):
            self.J = pg.load(dirname+"jacobian.bmat")
            print("Loaded jacobian: ", self.J.rows(), self.J.cols())

    def showResult(self, **kwargs):
        """Show inversion result."""
        kwargs.setdefault("logScale", True)
        kwargs.setdefault("cMap", "Spectral")
        kwargs.setdefault("xlabel", "x (m)")
        kwargs.setdefault("ylabel", "z (m)")
        kwargs.setdefault("label", r"$\rho$ ($\Omega$m)")
        return pg.show(self.mesh, 1./self.model, **kwargs)

    def exportRxTxVTK(self, marker=1):
        """Export Receiver and Transmitter positions as VTK file."""
        rxmesh = pg.Mesh(3)
        for i in range(self.nRx):
            rxmesh.createNode([self.rx[i], self.ry[i], self.rz[i]], marker)

        rxmesh.exportVTK(self.basename+"-rxpos.vtk")
        txmesh = pg.Mesh(3)
        for xx, yy in zip(self.tx, self.ty):
            txmesh.createNode(xx, yy, self.txAlt)

        for i in range(txmesh.nodeCount()-1):
            txmesh.createEdge(txmesh.node(i), txmesh.node(i+1), marker)

        txmesh.exportVTK(self.basename+"-txpos.vtk")

    def inversion(self,
                  inner_area_cell_size=1e4, outer_area_cell_size=None,  # m^2
                  inner_boundary_factor=.1, cell_size=1e7,  # m^3
                  invpoly=None, topo=None, useQHull=True, n_cores=60,
                  dim=None, extend_world=10, depth=1000., p_fwd=1,
                  tx_refine=50., rx_refine=30, tetgen_quality=1.3,
                  symlog_threshold=1e-4, sig_bg=0.001, **kwargs):
        """Run inversion including mesh generation etc.

        Does the whole inversion including pre- and post-processing:
        * check data and errors
        * automatical boundary computation
        * setting up meshes
        * run inversion parsing keyword arguments
        * load results
        * generate multipage pdf files showing data fit

        Parameters
        ----------
        Geometry

        depth : float [1000]
            Depth of the inversion region. The default is 1000..
        inner_area_cell_size : float [1e4]
            maximum cell size of inversion surface triangles in m^2.
        outer_area_cell_size : float [1e7]
            cell size of outer suface triangles in m^2.
        inner_boundary_factor : float
            Factor to add to the innerboundary. The default is .1 (=10%).
        cell_size : float
            Maximum tetrahedral cell size in m^3. The default is 1e7.
        invpoly : 2d-array, optional
            polygone for describing the shape of inversion domain.
        useQHull : bool [True]
            use convex hull for outer shape.
        topo : str
            topography file to by read. The default is None.
        extend_world : float
            extend world by a factor. The default is 10.
        tx_refine : float [30]
            tranmitter refinement in m. The default is 50..
        rx_refine : float [30]
            receiver refinement in m. The default is 30.
        tetgen_quality : float [1.3]
            Tetgen mesh quality. The default is 1.3.
        check : bool
            just make geometry, show it and quit (to optimize mesh pameters)

        Computation

        n_cores : int [60]
            Number of cores to use. The default is 60.
        dim : float
            Size of the modelling box. Auto-determined by default.
        p_fwd : int
            Polynomial order for forward computation. The default is 1.
        symlog_threshold : TYPE, optional
            Threshold for linear data transformation. The default is 1e-4.
        sig_bg : float [0.001]
            Background conductivity. The default is 0.001.
        **kwargs : dict
            Keyword arguments to be passed to inversion.
            lam : float
                regularization strength
            maxIter : int
                maximum iteration number
            robustData : bool [False]
                robust data fitting using an L1 norm
            blockyModel : bool
                enhance contrasts by using an L1 norm on roughness

        Plotting

        alim : (float, float) [1e-3, 1]
            limits for shwoing real and imaginary parts
        x : str ["y"]
            string indicating over which coordinate lines are plotted
        """
        if outer_area_cell_size is None:
            outer_area_cell_size = inner_area_cell_size * 100

        invmod = self.basename
        invmesh = 'invmesh_' + invmod
        dataname = self.basename or "mydata"

        if "npzfile" in kwargs:
            npzfile = kwargs.pop("npzfile", 'data/' + dataname + ".npz")
            saemdata = np.load(npzfile, allow_pickle=True)
        else:
            # %%
            saemdata = {}
            saemdata["DATA"], saemdata["line"] = self.getData(**kwargs)
            saemdata["tx"] = [np.column_stack([p.tx, p.ty, p.ty*0])
                              for p in self.patches]
            saemdata["origin"] = self.origin
            saemdata["rotation"] = self.angle
            saemdata["freqs"] = self.patches[0].f
            # cmp should not be needed as it is inside DATA
            saemdata["cmp"] = kwargs.setdefault("cmp", self.patches[0].cmp)
            # %%
        x0, y0 = 0, 0
        if invpoly is None:
            allrx = np.vstack([data["rx"][:, :2] for data in saemdata["DATA"]])
            alltx = np.vstack(saemdata["tx"])[:, :2]
            allrx = np.vstack([allrx, alltx])
            x0 = np.median(allrx[:, 0])
            y0 = np.median(allrx[:, 1])
            if useQHull:
                from scipy.spatial import ConvexHull
                points = allrx
                points -= [x0, y0]
                ch = ConvexHull(points)
                invpoly = np.array([[*points[v, :], 0.]
                                    for v in ch.vertices]) * \
                    (inner_boundary_factor + 1.0)
                invpoly += [x0, y0, 0.]
            else:
                xmin, xmax = min(allrx[:, 0]), max(allrx[:, 0])
                ymin, ymax = min(allrx[:, 1]), max(allrx[:, 1])
                dx = (xmax - xmin) * inner_boundary_factor
                dy = (ymax - ymin) * inner_boundary_factor
                invpoly = np.array([[xmin-dx, ymin-dy, 0.],
                                    [xmax+dx, ymin-dy, 0.],
                                    [xmax+dx, ymax+dy, 0.],
                                    [xmin-dy, ymax+dy, 0.]])

        if kwargs.pop("check", False):
            _, ax = self.showPositions()
            ax.plot(invpoly[:, 0], invpoly[:, 1], "k-")
            ax.plot(invpoly[::invpoly.shape[0]-1, 0],
                    invpoly[::invpoly.shape[0]-1, 1], "k-")
            return
        # generate npz structure as in saveData
        from custEM.meshgen.meshgen_tools import BlankWorld
        from custEM.meshgen import meshgen_utils as mu
        from custEM.inv.inv_utils import MultiFWD

        # extx = max(pg.x()
        # dim = dim or max()
        M = BlankWorld(name=invmesh,
                       x_dim=[x0-dim, x0+dim],
                       y_dim=[y0-dim, y0+dim],
                       z_dim=[-dim, dim],
                       preserve_edges=True,
                       t_dir='./',  # kann weg! lieber voller filename
                       topo=topo,
                       inner_area_cell_size=inner_area_cell_size,
                       easting_shift=-saemdata['origin'][0],
                       northing_shift=-saemdata['origin'][1],
                       rotation=float(saemdata['rotation'])*180/np.pi,
                       outer_area_cell_size=outer_area_cell_size,
                       )
        txs = [mu.refine_path(tx, length=tx_refine) for tx in saemdata['tx']]
        M.build_surface(insert_line_tx=txs)
        M.add_inv_domains(-depth, invpoly, cell_size=cell_size)
        M.build_halfspace_mesh()
        # %%
        # add receiver locations to parameter file for all receiver patches
        reducedrx = mu.resolve_rx_overlaps(
            [data["rx"] for data in saemdata["DATA"]], rx_refine)
        rx_tri = mu.refine_rx(reducedrx, rx_refine, 60.)
        M.add_paths(rx_tri)
        for rx in [data["rx"] for data in saemdata["DATA"]]:
            M.add_rx(rx)

        M.extend_world(extend_world, extend_world, extend_world)
        M.call_tetgen(tet_param='-pq{:f}aA'.format(tetgen_quality),
                      print_infos=False)

        alim = kwargs.pop("alim", [1e-3, 1])
        xy = kwargs.pop("x", "y")
        # setup fop
        fop = MultiFWD(invmod, invmesh, saem_data=saemdata, sig_bg=sig_bg,
                       n_cores=n_cores, p_fwd=p_fwd, start_iter=0)
        # fop.setRegionProperties("*", limits=[1e-4, 1])  # =>inv.setReg
        # set up inversion operator
        inv = pg.Inversion(fop=fop)
        inv.setPostStep(fop.analyze)
        dT = pg.trans.TransSymLog(symlog_threshold)
        inv.dataTrans = dT
        inv.setRegularization(limits=kwargs.pop("limits", [1e-4, 1.0]))
        # run inversion
        kwargs.setdefault("lam", 10)
        kwargs.setdefault("maxIter", 21)
        invmodel = inv.run(fop.measured, fop.errors, verbose=True,
                           startModel=fop.sig_0, **kwargs)
        # post-processing
        np.save(fop.inv_dir + 'inv_model.npy', invmodel)
        pgmesh = fop.mesh()
        pgmesh['sigma'] = invmodel
        pgmesh['res'] = 1. / invmodel
        cov = np.zeros(fop._jac.cols())
        mT = inv.modelTrans
        for i in range(fop._jac.rows()):
            cov += np.abs(fop._jac.row(i) * dT.deriv(inv.response) /
                          dT.error(inv.response, fop.errors))

        cov /= mT.deriv(invmodel)  # previous * invmodel
        cov /= pgmesh.cellSizes()

        np.save(fop.inv_dir + invmod + '_coverage.npy', cov)
        pgmesh['coverageLog10'] = np.log10(cov)
        pgmesh.exportVTK(fop.inv_dir + invmod + '_final_invmodel.vtk')
        resultdir = "inv_results/" + invmod + "_" + invmesh + "/"
        self.loadResults(dirname=resultdir)
        for i, p in enumerate(self.patches):
            p.generateDataPDF(resultdir+f"fit{i+1}.pdf",
                              mode="linefreqwise", x=xy, alim=alim)


if __name__ == "__main__":
    # %% way 1 - load ready patches
    self = CSEMSurvey()
    self.addPatch("Tx1.npz")
    self.addPatch("Tx2.npz")
    # %% way 2 - patch instances
    # patch1 = CSEMData("flight*.mat")
    # self.addPatch(patch1)
    # %% way 3 - directly from Mare file or npz
    # self = CSEMSurvey("blabla.emdata")
    # self = CSEMSurvey("blabla.npz")
    # %%
    print(self)
    self.showPositions()
    p = self.patches[0]
    # p.filter() etc.
    self.saveData()
