from p4p import Type, Value
from p4p.nt import NTTable
from p4p.rpc import rpc, quickRPCServer

name_table = NTTable([
  ("name","s")
])

class DirectoryService(object):
  @rpc(name_table)
  def ds(self, *args, **kws):
    responses = ["This", "is", "a", "test"]
    response_object = [{"name": s} for s in responses]
    return name_table.wrap(response_object)

print("Starting Directory Service Test Server!")
tester = DirectoryService()
quickRPCServer(provider="Directory Service Test Server",
               prefix="",
               target=tester)

"""
srv = pvaccess.RpcServer()
def handle_request(request):
	response = pvaccess.PvObject({"labels": [pvaccess.STRING], "value": {"name": [pvaccess.STRING]}}, {"labels": ["name"], "value": {"name": ['This', 'is', 'a', 'test']}},'epics:nt/NTTable:1.0')
	return response
"""
	
