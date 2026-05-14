import unittest
import subprocess
import sys
import meme.archive
from datetime import datetime, timedelta

class ArchiveGetTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.server_process = subprocess.Popen([sys.executable, '-m', 'tests.archive.test_pva_archive_server'])
      
  def test_single_pv_as_string(self):
    r = meme.archive.get("MC00:ASTS:OUTSIDET", protocol='PVA')
    self.assert_single_pv_response_has_correct_fields(r)
  
  def test_single_pv_as_list(self):
    r = meme.archive.get(["MC00:ASTS:OUTSIDET"], protocol='PVA')
    self.assert_single_pv_response_has_correct_fields(r)
    
  def test_multiple_pvs(self):
    pvs = ["MC00:ASTS:OUTSIDET", "QUAD:IN20:425:BDES"]
    r = meme.archive.get(pvs, protocol='PVA')
    self.assertTrue(isinstance(r, list))
    self.assertEqual(len(pvs), len(r))
    for i, item in enumerate(r):
      self.assert_multi_pv_response_has_correct_fields(item)
      self.assertEqual(item['pvName'],pvs[i])
    
  #These date arg tests are pretty dumb, all we really test
  #is that the get method doesn't completely barf when you give it
  #a string or a datetime.
  def test_date_args_as_strings(self):
    from_time_str = "1 hour ago"
    to_time_str = "30 minutes ago"
    r = meme.archive.get("MC00:ASTS:OUTSIDET", from_time=from_time_str, to_time=to_time_str, protocol='PVA')
    self.assert_single_pv_response_has_correct_fields(r)
  
  def test_date_args_as_datetimes(self):
    to_time = datetime.now()
    from_time = to_time - timedelta(hours=1)
    r = meme.archive.get("MC00:ASTS:OUTSIDET", from_time=from_time, to_time=to_time, protocol='PVA')
    self.assert_single_pv_response_has_correct_fields(r)

  def test_convert_to_dataframe_multiple_pvs(self):
    pvs = ["MC00:ASTS:OUTSIDET", "QUAD:IN20:425:BDES"]
    r = meme.archive.get(pvs, protocol='PVA')
    df = meme.archive.convert_to_dataframe(r)
    for pv in pvs:
      self.assertTrue(pv in df.keys())
  
  def test_convert_to_dataframe_single_pv(self):
    pvs = "MC00:ASTS:OUTSIDET"
    r = meme.archive.get(pvs, protocol='PVA')
    df = meme.archive.convert_to_dataframe(r)
    self.assertTrue(len(df.keys()) == 1)
    self.assertTrue(df.keys()[0] == "value")
  
  def assert_times_within_range(self, times, begin_time, end_time):
    begin_timestamp = (begin_time - datetime(1970,1,1)).total_seconds()
    end_timestamp = (end_time - datetime(1970,1,1)).total_seconds()
    self.assertTrue(min(times) >= begin_timestamp)
    self.assertTrue(max(times) <= end_timestamp)
    
  def assert_single_pv_response_has_correct_fields(self, response):
    self.assertTrue('status' in response)
    self.assertTrue('secondsPastEpoch' in response)
    self.assertTrue('values' in response)
    self.assertTrue('nanoseconds' in response)
    self.assertTrue('severity' in response)

  def assert_multi_pv_response_has_correct_fields(self, response):
    self.assertTrue('pvName' in response)
    self.assertTrue('value' in response)
    self.assert_single_pv_response_has_correct_fields(response['value']['value'])
  
  @classmethod
  def tearDownClass(cls):
    cls.server_process.terminate()
    cls.server_process.wait()