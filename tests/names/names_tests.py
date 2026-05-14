import unittest
import meme.names
import subprocess
import sys
import signal

class NameListTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.server_process = subprocess.Popen([sys.executable, '-m', 'tests.names.test_ds_server'])
	
	def setUp(self):
		self.expected_response = ["This", "is", "a", "test"]
				
	def test_list_pvs(self):
		l = meme.names.list_pvs("BPM:%", tag="LCLS", sort_by="z", element_type="INST")
		self.assertEqual(l, self.expected_response)
	
	def test_list_devices(self):
		l = meme.names.list_devices("BPM:%", tag="LCLS", sort_by="z", element_type="INST")
		self.assertEqual(l, self.expected_response)
	
	def test_list_elements(self):
		l = meme.names.list_elements("BPM:%", tag="LCLS", sort_by="z", element_type="INST")
		self.assertEqual(l, self.expected_response)
	
	@classmethod
	def tearDownClass(cls):
		cls.server_process.terminate()
		cls.server_process.wait()