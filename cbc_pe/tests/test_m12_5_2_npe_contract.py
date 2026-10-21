"""Tiny synthetic checks only: no data access, model training or GPU."""
import unittest
import numpy as np
import torch
from src.models.g1_npe import Coupling, to_physical, to_unconstrained, mass_bounds
from scripts.m12.analyze_m12_5_2_npe import bootstrap


class NPEContract(unittest.TestCase):
    def test_support_jacobian_and_order(self):
        z=torch.tensor([.3,-.4,.2],dtype=torch.float64,requires_grad=True)
        y=to_physical(z)
        recovered,ld,boundary=to_unconstrained(y)
        self.assertTrue(torch.allclose(z,recovered,atol=1e-10))
        self.assertFalse(bool(boundary))
        jac=torch.autograd.functional.jacobian(to_physical,z)
        self.assertAlmostEqual(float(torch.logdet(jac).detach()),float(-ld.detach()),places=9)
        lo,hi=mass_bounds(y[1])
        self.assertTrue(bool((y[0]>lo)&(y[0]<hi)&(y[2].abs()<1)))
        with self.assertRaises(ValueError):
            to_unconstrained(torch.tensor([100.,20.,0.]))

    def test_flow_inverse_and_finite_gradients(self):
        torch.manual_seed(12)
        layer=Coupling(1)
        with torch.no_grad():
            layer.net[-1].weight.normal_(std=.01)
        x=torch.randn(4,3,requires_grad=True);ctx=torch.randn(4,128)
        z,ld=layer(x,ctx);back,ild=layer(z,ctx,True)
        self.assertTrue(torch.allclose(back,x,atol=1e-6))
        self.assertTrue(torch.allclose(ld,-ild,atol=1e-6))
        (z.square().sum()+ld.sum()).backward()
        self.assertTrue(bool(torch.isfinite(x.grad).all()))

    def test_paired_fixed_environment_bootstrap(self):
        # Identical losses give exactly zero contrast under all cluster draws.
        values=np.zeros((12,3));env=np.array(['a']*8+['b']*4)
        for hierarchical in (False,True):
            out=bootstrap(values,env,20,np.random.default_rng(1),hierarchical)
            self.assertTrue(np.array_equal(out,np.zeros((20,3))))

if __name__=='__main__': unittest.main()
