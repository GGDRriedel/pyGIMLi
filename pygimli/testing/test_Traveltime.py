#!/usr/bin/env python
import unittest

import numpy as np
import pygimli as pg

from pygimli.physics import TravelTimeManager

class TestTT(unittest.TestCase):

    def setUp(self):
        # Dummy data container
        self.data = pg.DataContainer()
        self.data.createSensor([0.0, 0.0])
        self.data.createSensor([1.0, 2.0])
        self.data.resize(1)
        self.data.set("s", [1])
        self.data.set("g", [2])
        self.data.registerSensorIndex("s")
        self.data.registerSensorIndex("g")

        # Without secondary nodes
        self.mesh = pg.createGrid([0,1,2],[0,1,2])

        # Slowness
        self.slo = [1,2,1,4]
        
        self.mgr = TravelTimeManager()

    def test_withoutSecNodes(self):
        fop = self.mgr.fop
        fop.setData(self.data)
        fop.setMesh(self.mesh, 
                    ignoreRegionManager=True)
        t = fop.response(self.slo)
        np.testing.assert_allclose(t, 1 + np.sqrt(2))

        pg.show(self.mesh)
        data = self.mgr.simulate(slowness=self.slo, scheme=self.data, 
                                 mesh=self.mesh, secNodes=0)
        np.testing.assert_allclose(data['t'], 1 + np.sqrt(2))
        

    def test_withSecNodes(self):
        fop = self.mgr.fop
        fop.setData(self.data)
        fop.setMesh(self.mesh.createMeshWithSecondaryNodes(n=3), 
                    ignoreRegionManager=True)

        t = fop.response(self.slo)
        np.testing.assert_allclose(t, np.sqrt(5)) # only works for odd secNodes
        
        data = self.mgr.simulate(slowness=self.slo, scheme=self.data, 
                                 mesh=self.mesh, secNodes=3)
        np.testing.assert_allclose(data['t'], np.sqrt(5)) # only works for odd secNodes


    def test_Jacobian(self):
        fop = self.mgr.fop
        fop.setData(self.data)
        fop.setMesh(self.mesh.createMeshWithSecondaryNodes(n=5), 
                    ignoreRegionManager=True)

        fop.createJacobian(self.slo)
        J = fop.jacobian()
        np.testing.assert_allclose(J * self.slo, np.sqrt(5))

    def test_CudaAvailability(self):
        """Test CUDA availability detection."""
        # Should not raise an error, just return True or False
        cuda_available = TravelTimeManager.cudaAvailable()
        self.assertIsInstance(cuda_available, bool)
    
    @unittest.skipIf(not TravelTimeManager.cudaAvailable(), "CUDA not available")
    def test_CudaComputation(self):
        """Test CUDA computation produces same results as CPU."""
        # Create a manager with CUDA enabled
        mgr_cuda = TravelTimeManager(useCuda=True)
        mgr_cuda.fop.setData(self.data)
        mgr_cuda.fop.setMesh(self.mesh, ignoreRegionManager=True)
        
        # Compute with CUDA
        t_cuda = mgr_cuda.fop.response(self.slo)
        
        # Create a manager with CPU only
        mgr_cpu = TravelTimeManager(useCuda=False)
        mgr_cpu.fop.setData(self.data)
        mgr_cpu.fop.setMesh(self.mesh, ignoreRegionManager=True)
        
        # Compute with CPU
        t_cpu = mgr_cpu.fop.response(self.slo)
        
        # Results should be identical (or very close)
        np.testing.assert_allclose(t_cuda, t_cpu, rtol=1e-10)
    
    def test_CudaFallback(self):
        """Test CUDA fallback to CPU when CUDA is not available."""
        # This should not raise an error even if CUDA is not available
        mgr = TravelTimeManager(useCuda=True)
        mgr.fop.setData(self.data)
        mgr.fop.setMesh(self.mesh, ignoreRegionManager=True)
        
        # Computation should work (using CPU if CUDA not available)
        t = mgr.fop.response(self.slo)
        np.testing.assert_allclose(t, 1 + np.sqrt(2))
    
    def test_CudaToggle(self):
        """Test enabling and disabling CUDA at runtime."""
        mgr = TravelTimeManager()
        
        # Initially should be disabled
        self.assertFalse(mgr.useCuda())
        
        # Enable CUDA
        mgr.setUseCuda(True)
        
        # Check the state (might still be False if CUDA not available)
        cuda_state = mgr.useCuda()
        self.assertIsInstance(cuda_state, bool)
        
        # Disable CUDA
        mgr.setUseCuda(False)
        self.assertFalse(mgr.useCuda())

if __name__ == '__main__':

    # fop  = TestTT()
    # fop.test_MT()
    
    unittest.main()