import unittest
from meme.model import Model
import numpy as np
import subprocess
import sys

class ModelTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.server_process = subprocess.Popen([sys.executable, '-m', 'tests.model.model_mailbox'])
    
  def test_get_rmat_single_element(self):
    m = Model("TEST")
    self.assertEqual(m.get_rmat("XCOR:IN20:112").shape, (6,6))
    
  def test_get_rmat_one_list(self):
    m = Model("TEST")
    dev_list = ["SOL:IN20:111", "XCOR:IN20:112", "YCOR:IN20:113"]
    rs = m.get_rmat(dev_list)
    self.assertEqual(rs.shape, (len(dev_list),6,6))
  
  def test_get_rmat_from_a_to_list(self):
    m = Model("TEST")
    dev_list = ["SOL:IN20:111", "XCOR:IN20:112", "YCOR:IN20:113"]
    rs = m.get_rmat("XCOR:IN20:112", dev_list)
    self.assertEqual(rs.shape, (len(dev_list),6,6))
  
  def test_get_rmat_from_list_to_b(self):
    m = Model("TEST")
    dev_list = ["SOL:IN20:111", "XCOR:IN20:112", "YCOR:IN20:113"]
    rs = m.get_rmat(dev_list, "YCOR:IN20:113")
    self.assertEqual(rs.shape, (len(dev_list),6,6))
  
  def test_get_rmat_from_list_to_list(self):
    m = Model("TEST")
    a_list = ["SOL:IN20:111", "XCOR:IN20:112"]
    b_list = ["XCOR:IN20:112", "YCOR:IN20:113"]
    rs = m.get_rmat(a_list, b_list)
    self.assertEqual(rs.shape, (len(a_list),6,6))
  
  def test_get_zpos_single_element(self):
    m = Model("TEST")
    self.assertTrue(isinstance(m.get_zpos("XCOR:IN20:112"), float))
  
  def test_get_zpos_list(self):
    m = Model("TEST")
    dev_list = ["SOL:IN20:111", "XCOR:IN20:112", "YCOR:IN20:113"]
    zs = m.get_zpos(dev_list)
    self.assertEqual(len(dev_list), len(zs))
  
  def test_get_twiss_single_element(self):
    m = Model("TEST")
    t = m.get_twiss("XCOR:IN20:112")
    self.assert_has_twiss_fields(t)
  
  def test_get_twiss_list(self):
    m = Model("TEST")
    dev_list = ["SOL:IN20:111", "XCOR:IN20:112", "YCOR:IN20:113"]
    ts = m.get_twiss(dev_list)
    self.assertEqual(len(dev_list), len(ts))
  
  def assert_has_twiss_fields(self, t):
    names = t.dtype.names
    self.assertTrue('length' in names)
    self.assertTrue('p0c' in names)
    self.assertTrue('psi_x' in names)
    self.assertTrue('beta_x' in names)
    self.assertTrue('alpha_x' in names)
    self.assertTrue('eta_x' in names)
    self.assertTrue('etap_x' in names)
    self.assertTrue('psi_y' in names)
    self.assertTrue('beta_y' in names)
    self.assertTrue('alpha_y' in names)
    self.assertTrue('eta_y' in names)
    self.assertTrue('etap_y' in names)
  
  @classmethod
  def tearDownClass(cls):
    cls.server_process.terminate()
    cls.server_process.wait()