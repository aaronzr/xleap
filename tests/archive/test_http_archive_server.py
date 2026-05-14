import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
import json

PORT = 8000

class ArchiveHandler(http.server.SimpleHTTPRequestHandler):
	def do_GET(self):
		# Custom logic for GET requests
		
		parsed_url = urlparse(self.path)
		query_params = parse_qs(parsed_url.query)
		pv = query_params.get('pv')
		response = json.dumps([
			{"meta": {"name": pv[0], "EGU": "DegF", "PREC": "1"},
			"data": [
				{ "secs": 1759314206, "val": 65.39, "nanos": 492782992, "severity":0, "status":0 },
			]}
		]).encode('utf-8')
		
		
		self.send_response(200)
		self.send_header("Content-type", "text/html")
		self.end_headers()
		self.wfile.write(response)

with socketserver.TCPServer(("", PORT), ArchiveHandler) as httpd:
	print("serving at port", PORT)
	httpd.serve_forever()