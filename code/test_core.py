import unittest,json,sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from scipy.optimize import minimize
from scipy.stats import t,norm
from core import *

class RegressionTests(unittest.TestCase):
 def test_zero_scale_counterexample(self):
  for R in (1.,10.,1e6):
   x=np.r_[np.zeros(5),np.full(3,R)]
   for loss in ('pseudo','canonical'):
    fit=adaptive(x,2,loss);self.assertEqual(fit.branch,'zero_scale');self.assertEqual(fit.center[0],0.)
   self.assertAlmostEqual(canonical_huber(x,2*R).center[0]/R,3/8,places=10)
   self.assertAlmostEqual(pseudo_huber(x,2*R).center[0]/R,.367764,places=6)
 def test_coincident_nonmedian(self):
  x=np.array([[0.,0.],[1.,0.],[2.,0.]])
  f=geometric_median(x,initial=x[0]);self.assertTrue(f.converged);np.testing.assert_allclose(f.center,[1.,0.],atol=1e-8)
 def test_repeated_collinear(self):
  x=np.array([[0.,0.]]*5+[[9.,0.]]*3)
  f=geometric_median(x);np.testing.assert_array_equal(f.center,[0.,0.]);self.assertEqual(f.branch,'exact_majority')
 def test_quadratic_and_large_threshold(self):
  x=np.array([[0.,1.],[1.,0.],[2.,2.]])
  np.testing.assert_allclose(canonical_huber(x,10).center,x.mean(0),atol=1e-12)
  np.testing.assert_allclose(pseudo_huber(x,1e6).center,x.mean(0),atol=1e-10)
 def test_covariance_scaling_and_singular(self):
  z=np.array([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]])
  V,d=projected_sandwich(z,[0,0],1e6,np.eye(2));self.assertTrue(d['singular'])
  self.assertAlmostEqual(V[0,0]/4,np.var(z[:,0])/4,places=10)
  # Original summaries Z=W/sqrt(m), tau=lambda/sqrt(m): covariance scales 1/m.
  m=13;V2,_=projected_sandwich(z/np.sqrt(m),[0,0],1e6/np.sqrt(m),np.eye(2))
  np.testing.assert_allclose(V2,V/m,atol=1e-12)
 def test_exact_targets(self):
  for m,u in [(8,-.0919315),(32,-.0480795),(128,-.0237467),(512,-.0118386)]:
   result=exact_binomial_target(m,.1);self.assertAlmostEqual(result['u'],u,places=7)
   self.assertLess(result['score_residual'],1e-12)
 def test_independent_optimizer(self):
  x=np.random.default_rng(21).normal(size=(15,3));fit=pseudo_huber(x,.8)
  fun=lambda v:np.mean(.8**2*(np.sqrt(1+np.sum((x-v)**2,axis=1)/.8**2)-1))
  opt=minimize(fun,x.mean(0),method='BFGS',tol=1e-11)
  self.assertLess(np.linalg.norm(fit.center-opt.x),2e-7)
  self.assertLess(fit.certificate,1e-7)
  gm=geometric_median(x)
  gmopt=minimize(lambda v:np.linalg.norm(x-v,axis=1).mean(),x.mean(0),method='BFGS',tol=1e-10)
  self.assertLess(np.linalg.norm(gm.center-gmopt.x),2e-6)
 def test_equivariance(self):
  x=np.random.default_rng(2).normal(size=(20,3));q=np.linalg.qr(np.random.default_rng(3).normal(size=(3,3)))[0]
  f=adaptive(x);g=adaptive(3*x@q+np.array([2.,1.,-4.]))
  np.testing.assert_allclose(g.center,3*f.center@q+[2,1,-4],atol=1e-7)
 def test_nonfinite(self):
  with self.assertRaises(ValueError): pseudo_huber([[np.nan]],1.)
 def test_gaussian_endpoint(self):
  expected=[.8906,.9229,.9371,.9437]
  for k,v in zip([8,16,32,64],expected):
   coverage=2*t.cdf(norm.ppf(.975)*np.sqrt((k-1)/k),k-1)-1
   self.assertAlmostEqual(coverage,v,places=4)
 def test_wearable_raw_window_energy(self):
  import run_wearable
  # Opposite signs within windows distinguish mean squared energy from the
  # squared mean signal. Outputs stay in a temporary directory.
  with TemporaryDirectory() as temporary:
   root=Path(temporary);output=root/'derived';output.mkdir()
   fixtures={'train':([1,1,2],[1,1,2],[[1,-1],[3,-3],[2,2]],[[2,2],[0,0],[0,0]],[[0,0],[4,4],[0,0]]),
             'test':([3,3],[1,2],[[1,1],[0,0]],[[1,-1],[1,1]],[[1,1],[0,0]])}
   for split,(subjects,activities,x,y,z) in fixtures.items():
    signals=root/split/'Inertial Signals';signals.mkdir(parents=True)
    np.savetxt(root/split/f'subject_{split}.txt',subjects,fmt='%d')
    np.savetxt(root/split/f'y_{split}.txt',activities,fmt='%d')
    for axis,values in zip('xyz',(x,y,z)):
     np.savetxt(signals/f'body_acc_{axis}_{split}.txt',values)
   with patch.object(run_wearable,'DATA',output):
    derived=run_wearable.preprocess(root).set_index(['subject','activity'])
   self.assertEqual(derived.loc[(1,1),'energy'],15.)
   self.assertEqual(derived.loc[(1,1),'n_windows'],2)
   self.assertEqual(derived.loc[(2,2),'energy'],4.)
   self.assertEqual(derived.loc[(3,1),'energy'],3.)
   self.assertEqual(derived.loc[(3,2),'energy'],1.)
   self.assertTrue((output/'subject_window_energy.csv').is_file())
   manifest=json.loads((output/'preprocessing_manifest.json').read_text())
   self.assertEqual(len(manifest['input_sha256']),6)

if __name__=='__main__':unittest.main(verbosity=2)
