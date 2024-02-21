"""Controlled-source electromagnetic (CSEM) survey (patch collection) data."""
import os.path
from glob import glob

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

import pygimli as pg
from pygimli.core.math import symlog

from saem import Mare2dEMData
from saem import CSEMData
from saem.mt import MTData
from saem.tools import coverage


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

    def loadNPZ(self, filename, mtdata=False, mode=None, **kwargs):
        """Load numpy-compressed (NPZ) file."""
        ALL = np.load(filename, allow_pickle=True)
        self.f = ALL["freqs"]
        self.origin = ALL["origin"]
        if "tx" in ALL:
            self.tx = ALL["tx"]
        a = 0
        if "line" in list(ALL.keys()):
            line = ALL["line"]
        else:
            pass
        line = np.array([], dtype=int)
        for data in ALL["DATA"]:
            line = np.append(line, np.ones(len(data['rx']), dtype=int))

        if mode is None:
            if not mtdata:
                mode = 'B'
            else:
                mode = 'T'

        for i in range(len(ALL["DATA"])):
            if 'E' in mode or 'B' in mode:
                patch = CSEMData(mode=mode)
            else:
                patch = MTData(mode=mode)

            patch.extractData(ALL, i)
            self.addPatch(patch)

            if len(line) > 0:
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
        # this part should be moved into Mare2dEMData
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

    def addPatch(self, patch, name=None, **kwargs):
        """Add a new patch to the file.

        Parameters
        ----------
        patch : CSEMData | str
            CSEMData instance or string to load into that
        """
        if isinstance(patch, str):
            patch = CSEMData(patch, **kwargs)
            if name is not None:
                patch.basename = name

        if len(self.patches) == 0:
            self.angle = patch.angle
            self.origin = patch.origin
            self.cmp = patch.cmp
            # print('  - copy *angle*, *origin* and *cmp* from first patch  -')
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
        """Show all positions in one plot."""
        ax = kwargs.pop("ax", None)
        if ax is None:
            _, ax = plt.subplots()

        ma = ["x", "+", "^", "v"]
        for i, p in enumerate(self.patches):
            p.showPositions(ax=ax, color="C{:d}".format(i),
                            marker=ma[i % len(ma)], **kwargs)

        return ax

    def showData(self, **kwargs):
        """Show data of individual patches."""
        for p in self.patches:
            p.showData(**kwargs)

    def setOrigin(self, *args, **kwargs):
        """Set the same origin for all patches (reshifting if existing)."""
        for p in self.patches:
            p.setOrigin(*args, **kwargs)

    def rotate(self, *args, **kwargs):
        """SRotate all patches."""
        for p in self.patches:
            p.rotate(*args, **kwargs)

    def detectLines(self, *args, **kwargs):
        """SRotate all patches."""
        for p in self.patches:
            p.detectLines(*args, **kwargs)

    def filter(self, *args, **kwargs):
        """Filter."""
        for p in self.patches:
            p.filter(*args, **kwargs)

    def estimateError(self, *args, **kwargs):
        """Estimate error model."""
        mask = kwargs.pop("mask", True)
        for p in self.patches:
            p.estimateError(*args, **kwargs)
            if mask:
                p.deactivateNoisyData()

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
        """Ensure that old scripts with the old method name work."""
        self.createDataDict(fname=fname, save=True, line=line, **kwargs)

    def createDataDict(self, fname=None, save=False, line=None, **kwargs):
        """Create saemdata Dict and save data in numpy format for inversion."""
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
        self.DDict = {'tx' : txs,
                      'freqs' : self.patches[0].f,
                      'DATA' : DATA,
                      'line' : lines,
                      'cmp' : [patch["cmp"] for patch in DATA],
                      'origin' : self.origin,
                      'rotation' : self.angle}
        if save:
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
            # respfiles = sorted(glob(dirname+"response_iter*.npy")) # bug 9>11
            respfiles = glob(dirname+"response_iter*.npy")
            if len(respfiles) == 0:
                pg.error("Could not find response file")
            sI = np.argsort([int(respfile.replace(dirname, '')[14:-4])
                             for respfile in respfiles])
            respfiles = [respfiles[i] for i in sI]
        else:
            if isinstance(response, str):
                respfiles = [response]
            elif isinstance(response, int):
                respfiles = [dirname + "response_iter_" + str(response) + ".npy"]

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

    def showJacobianRow(self, iI=1, iC=0, iF=0, iR=0, cM=1.5, tol=1e-7,
                        save=False, **kwargs):
        """Show Jacobian row (model distribution for specific data).

        Parameters
        ----------
        iI : int [1]
            real (0) or imaginary (1) part
        iC : int
            component (out of existing ones!)
        iF : int [0]
            frequency index (into self.f)
        iR : int [0]
            receiver number
        cM : float [1.5]
            color scale maximum
        tol : float
            tolerance/threshold for symlog transformation

        **kwargs are passed to ps.show (e.g. cMin, )
        """
        allcmp = ['Bx', 'By', 'Bz']
        scmp = [allcmp[i] for i in np.nonzero(self.cmp)[0]]
        allP = ["Re ", "Im "]
        nD = self.J.rows() // 2
        nC = sum(self.cmp)
        nF = self.patches[0].nF
        nR = sum([p.nRx for p in self.patches])
        assert nD == nC * nF * nR, "Dimensions mismatch"
        iD = nD*iI + iC*(nF*nR) + iF*nR + iR
        Jrow = self.J.row(iD)
        sens = symlog(Jrow / self.mesh.cellSizes() * self.model, tol=tol)
        defaults = dict(cMap="bwr", cMin=-cM, cMax=cM, colorBar=False,
                        xlabel="x (m)", ylabel="z (m)")
        defaults.update(kwargs)
        ax, cb = pg.show(self.mesh, sens, **defaults)
        ax.plot(np.mean(self.tx), 5, "k*")
        ax.plot(self.patches[0].rx[iR], 10, "kv")
        st = allP[iI] + scmp[iC] + ", f={:.0f}Hz, x={:.0f}m".format(
            self.f[iF], self.patches[0].rx[iR])
        ax.set_title(st)
        fn = st.replace(" ", "_").replace(",", "").replace("=", "")
        if save:
            ax.figure.savefig("pics/"+fn+".pdf", bbox_inches="tight")

        return ax

    def showResult(self, **kwargs):
        """Show inversion result."""
        kwargs.setdefault("logScale", True)
        kwargs.setdefault("cMap", "Spectral")
        kwargs.setdefault("xlabel", "x (m)")
        kwargs.setdefault("ylabel", "z (m)")
        kwargs.setdefault("label", r"$\rho$ ($\Omega$m)")
        return pg.show(self.mesh, 1./self.model, **kwargs)

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
        ........
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
        ...........
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
        ........
        alim : (float, float) [1e-3, 1]
            limits for shwoing real and imaginary parts
        x : str ["y"]
            string indicating over which coordinate lines are plotted
        """
        if outer_area_cell_size is None:
            outer_area_cell_size = inner_area_cell_size * 100

        invmod = self.basename
        invmesh=kwargs.pop('invmesh','invmesh_' + invmod)
        # invmesh = 'invmesh_' + invmod
        dataname = self.basename or "mydata"

        if "npzfile" in kwargs:
            npzfile = kwargs.pop("npzfile", 'data/' + dataname + ".npz")
            saemdata = np.load(npzfile, allow_pickle=True)
        else:
            saemdata = {}
            saemdata["DATA"], saemdata["line"] = self.getData(**kwargs)
            saemdata["tx"] = [np.column_stack([p.tx, p.ty, p.ty*0])
                              for p in self.patches]
            saemdata["origin"] = self.origin
            saemdata["rotation"] = self.angle
            saemdata["freqs"] = self.patches[0].f
            # cmp should not be needed as it is inside DATA
            saemdata["cmp"] = kwargs.setdefault("cmp", self.patches[0].cmp)

        x0, y0 = 0, 0
        if invpoly is None:
            allrx = np.vstack([data["rx"][:, :2] for data in saemdata["DATA"]])
            alltx = np.vstack(saemdata["tx"])[:, :2]
            points = np.vstack([allrx, alltx])
            x0 = np.median(allrx[:, 0])
            y0 = np.median(allrx[:, 1])
            if useQHull:
                points -= [x0, y0]
                ch = ConvexHull(points)
                invpoly = np.array([[*points[v, :], 0.]
                                    for v in ch.vertices]) * \
                    (inner_boundary_factor + 1.0)
                invpoly += [x0, y0, 0.]
            else:
                xmin, xmax = min(points[:, 0]), max(points[:, 0])
                ymin, ymax = min(points[:, 1]), max(points[:, 1])
                dx = (xmax - xmin) * inner_boundary_factor
                dy = (ymax - ymin) * inner_boundary_factor
                invpoly = np.array([[xmin-dx, ymin-dy, 0.],
                                    [xmax+dx, ymin-dy, 0.],
                                    [xmax+dx, ymax+dy, 0.],
                                    [xmin-dy, ymax+dy, 0.]])

        ext = max(max(invpoly[:, 0]) - min(invpoly[:, 0]),
                  max(invpoly[:, 1]) - min(invpoly[:, 1]))
        dim = dim or ext*5
        print("xdim: ", [x0-dim, x0+dim])
        print("ydim: ", [y0-dim, y0+dim])

        if kwargs.pop("check", False):
            ax = self.showPositions()
            ax.plot(invpoly[:, 0], invpoly[:, 1], "k-")
            ax.plot(invpoly[::invpoly.shape[0]-1, 0],
                    invpoly[::invpoly.shape[0]-1, 1], "k-")
            return
        # generate npz structure as in saveData
        from custEM.meshgen.meshgen_tools import BlankWorld
        from custEM.meshgen import meshgen_utils as mu
        from custEM.inv.inv_utils import MultiFWD
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

        # add receiver locations to parameter file for all receiver patches
        reducedrx = mu.resolve_rx_overlaps(
            [data["rx"] for data in saemdata["DATA"]], rx_refine)
        rx_tri = mu.refine_rx(reducedrx, rx_refine, 30.)
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
                       n_cores=n_cores, p_fwd=p_fwd)
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
        kwargs.setdefault('startModel', fop.sig_0)
        invmodel = inv.run(fop.measured, fop.errors, verbose=True,
                           **kwargs)
        # post-processing
        np.save(fop.inv_dir + 'inv_model.npy', invmodel)
        pgmesh = fop.mesh()
        pgmesh['sigma'] = invmodel
        pgmesh['res'] = 1. / invmodel
        cov = np.zeros(fop._jac.cols())
        mT = inv.modelTrans
        dataScale = dT.deriv(inv.response) / \
            dT.error(inv.response, fop.errors)
        for i in range(fop._jac.rows()):
            cov += np.abs(fop._jac.row(i) * dataScale[i])

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

    # old inversion() - differences:
    # inner_area_cell_size=1e4, outer_area_cell_size=None,  # m^2
    # cell_size=1e7, # m^3
    # invpoly=None, useQHull=True, => only invpoly(array|'Qhull' else rect)
    # tx_refine=50. => 10, rx_refine=30 => 10,
    def buildInvMesh(self,
                invmesh=None, depth=1000., surface_cz=1e4,
                inner_boundary_factor=0.1, inv_cz=1e7, dim=None,
                invpoly='Qhull', topo=None, check_pos=True,
                extend_world=10., tx_refine=10., rx_refine=10,
                tetgen_quality=1.3, **kwargs):
        """Mesh generation before inversion

        Parameters
        ----------
        depth : float [1000]
            Depth of the inversion region. The default is 1000
        surface_cz : float [1e4]
            Maximum cell size of inversion surface triangles in m^2
        inner_boundary_factor : float
            Factor to add to the innerboundary. The default is .1 (=10%)
        inv_cz : float
            Maximum tetrahedral cell size in m^3. The default is 1e7
        invpoly : str [Qhull] or 2d-array specifying polygone
            Polygone for describing the shape of inversion domain
        topo : str
            Topography file to by read. The default is None
        extend_world : float
            Extend world by a factor. The default is 10
        tx_refine : float [10]
            Tranmitter refinement in m. The default is 50
        rx_refine : float [10]
            Receiver refinement in m. The default is 30
        tetgen_quality : float [1.3]
            Tetgen mesh quality. The default is 1.3
        check_pos: bool [True]
            Show Rx and Tx postions before calling TetGen
        **kwargs : dict
            Other keyword arguments that can be passed to set meshing options
        """
        if not hasattr(self, 'Ddict'):
            self.createDataDict()

        if invmesh is None:
            invmesh = self.basename + '_mesh'

        x0, y0 = 0, 0
        if isinstance(invpoly, str):
            allrx = np.vstack([d["rx"][:, :2] for d in self.DDict["DATA"]])
            alltx = np.vstack(self.DDict["tx"])[:, :2]
            points = np.vstack([allrx, alltx])
            x0 = np.median(allrx[:, 0])
            y0 = np.median(allrx[:, 1])
            if invpoly == 'Qhull':
                points -= [x0, y0]
                ch = ConvexHull(points)
                invpoly = np.array([[*points[v, :], 0.]
                                    for v in ch.vertices]) * \
                    (inner_boundary_factor + 1.0)
                invpoly += [x0, y0, 0.]
            else:
                xmin, xmax = min(points[:, 0]), max(points[:, 0])
                ymin, ymax = min(points[:, 1]), max(points[:, 1])
                dx = (xmax - xmin) * inner_boundary_factor
                dy = (ymax - ymin) * inner_boundary_factor
                invpoly = np.array([[xmin-dx, ymin-dy, 0.],
                                    [xmax+dx, ymin-dy, 0.],
                                    [xmax+dx, ymax+dy, 0.],
                                    [xmin-dy, ymax+dy, 0.]])
        ext = max(max(invpoly[:, 0]) - min(invpoly[:, 0]),
                  max(invpoly[:, 1]) - min(invpoly[:, 1]))
        dim = dim or ext*5
        kwargs.setdefault("x_dim", [x0-dim, x0+dim])
        kwargs.setdefault("y_dim", [y0-dim, y0+dim])
        kwargs.setdefault("z_dim", [-dim, dim])
        if check_pos:
            ax = self.showPositions()
            ax.plot(invpoly[:, 0], invpoly[:, 1], "k-")
            ax.plot(invpoly[::invpoly.shape[0]-1, 0],
                    invpoly[::invpoly.shape[0]-1, 1], "k-")

        # generate npz structure as in saveData
        from custEM.meshgen.meshgen_tools import BlankWorld
        from custEM.meshgen import meshgen_utils as mu

        M = BlankWorld(name=invmesh,
                       preserve_edges=True,
                       topo=topo,
                       inner_area_cell_size=surface_cz,
                       easting_shift=-self.DDict['origin'][0],
                       northing_shift=-self.DDict['origin'][1],
                       rotation=float(self.DDict['rotation'])*180/np.pi,
                       **kwargs,
                       )
        txs = [mu.refine_path(tx, length=tx_refine) for tx in self.DDict['tx']]
        M.build_surface(insert_line_tx=txs)
        M.add_inv_domains(-depth, invpoly, cell_size=inv_cz)
        M.build_halfspace_mesh()

        # add receiver locations to parameter file for all receiver patches
        rxs = mu.resolve_rx_overlaps(
            [data["rx"] for data in self.DDict["DATA"]], rx_refine)
        rx_tri = mu.refine_rx(rxs, rx_refine, 30.)
        M.add_paths(rx_tri)
        for rx in [data["rx"] for data in self.DDict["DATA"]]:
            M.add_rx(rx)

        M.extend_world(extend_world, extend_world, extend_world)
        M.call_tetgen(tet_param='-pq{:f}aA'.format(tetgen_quality),
                      print_infos=False)

    def runInv(self, invmesh=None,
               sig_bg=0.001, n_cores=72, p_fwd=1, symlog_threshold=0,
               make_plots=True, saem_data=None, invmod=None,
               lam=1., lamFactor=0.8, maxIter=21, robustData=False,
               blockyModel=False, **kwargs):

        """Run inversion

        Does inversion includingpost-processing:
        * run inversion
        * load results
        * generate multipage pdf files showing data fit

        Parameters
        ----------
        Computation
        ...........
        n_cores : int [60]
            Number of cores to use. The default is 60.
        p_fwd : int [1]
            Polynomial order for forward computation. The default is 1.
        symlog_threshold : float [None]
            If specified, a symlog data transformation will be used with the
            given threshold for the linear scale.
        sig_bg : float [0.001]
            Background conductivity. The default is 0.001.
        make_plots : bool [True]
            Make plots automatically after successful inversion run
        saem_data : dictionary [None]
            Use independent saem data dictionary
        invmod : str [None]
            Specify name for inversion run
        lam : float
            Regularization strength
        lamFactor : float
            Factor for decreasing lambda in each iteration
        maxIter : int
            Maximum iteration number
        robustData : bool [False]
            Robust data fitting using an L1 norm
        blockyModel : bool
            Enhance contrasts by using an L1 norm on roughness
        **kwargs : dict
            Other keyword arguments that can be passed to the inversion call

        Plotting
        ........
        alim : (float, float) [1e-3, 1]
            limits for shwoing real and imaginary parts
        x : str ["y"]
            string indicating over which coordinate lines are plotted
        """

        # usually, this should not be reuiqred here, maybe replace with a more
        # elaborate check if mesh exists or something
        if invmesh is None:
            invmesh = self.basename + '_mesh'
        if not hasattr(self, 'Ddict'):
            self.createDataDict()
        if saem_data is None:
            saem_data = self.DDict
        if invmod is None:
            invmod = self.basename

        # setup fop
        from custEM.inv.inv_utils import MultiFWD
        fop = MultiFWD(invmod, invmesh, saem_data=saem_data, sig_bg=sig_bg,
                       n_cores=n_cores, p_fwd=p_fwd)
        # fop.setRegionProperties("*", limits=[1e-4, 1])  # =>inv.setReg
        # set up inversion operator
        inv = pg.Inversion(fop=fop)
        inv.setRegularization(limits=kwargs.pop("limits", [1e-4, 1.0]))
        inv.setPostStep(fop.analyze)
        if symlog_threshold > 0:  # otherwise linear
            dT = pg.trans.TransSymLog(symlog_threshold)
            inv.dataTrans = dT

        # run inversion
        kwargs.setdefault('startModel', fop.sig_0)
        invmodel = inv.run(fop.measured, relativeError=fop.errors, verbose=True, lam=lam,
                           lamFactor=lamFactor, maxIter=maxIter,
                           robustData=robustData, blockModel=blockyModel,
                           **kwargs)
        # post-processing
        np.save(fop.inv_dir + 'inv_model.npy', invmodel)
        pgmesh = fop.mesh()
        pgmesh['sigma'] = invmodel
        pgmesh['res'] = 1. / invmodel
        pgmesh['coverage'] = coverage(inv)#, invmodel)
        pgmesh['coverageLog10'] = np.log10(pgmesh['coverage'])
        pgmesh.exportVTK(fop.inv_dir + invmod + '_final_invmodel.vtk')

        # plotting
        if make_plots:
            self.loadResults(dirname=fop.inv_dir)
            alim = kwargs.pop("alim", [1e-3, 1])
            xy = kwargs.pop("x", "y")
            for i, p in enumerate(self.patches):
                p.generateDataPDF(fop.inv_dir+f"fit{i+1}.pdf",
                                  mode="linefreqwise", x=xy, alim=alim)
                p.generateDataPDF(fop.inv_dir+f"wmisfit{i+1}.pdf",
                                  mode="patchwise", log=False)


if __name__ == "__main__":
    # %% way 1 - load ready patches
    survey = CSEMSurvey()
    survey.addPatch("Tx1.npz")
    survey.addPatch("Tx2.npz")
    # %% way 2 - patch instances
    # patch1 = CSEMData("flight*.mat")
    # self.addPatch(patch1)
    # %% way 3 - directly from Mare file or npz
    # self = CSEMSurvey("blabla.emdata")
    # self = CSEMSurvey("blabla.npz")
    # %%
    print(survey)
    survey.showPositions()
    patch = survey.patches[0]  # or self[0]
    # p.filter() etc.
    survey.saveData()
