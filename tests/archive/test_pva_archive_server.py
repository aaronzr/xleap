from p4p import Type, Value
from p4p.nt import NTTable
from p4p.rpc import rpc, quickRPCServer

single_pv_struct = NTTable([("secondsPastEpoch", "l"), ("values", "d"), ("nanoseconds", "i"), ("severity", "i"), ("status", "i")])

"""
pv_data = pvaccess.PvObject({"value": {"pvName": pvaccess.STRING, "value": single_pv_struct}},{"value": {"pvName": pv, "value": single_pv_value}})
"""

multi_pv_struct = Type([
  ("pvName", "s"),
  ("value", ("S", "NTComplexTable", single_pv_struct.type.items()))
])

multi_response_struct = Type([
  ("value", "av")
])

class ArchiveTester(object):
  @rpc(None)
  def hist(self, **kws):
    pv = kws['pv']
    value = [{"secondsPastEpoch": 1, "values": 123.45, "nanoseconds": 0, "severity": 0, "status": 0}]
    pv_list = pv.split(",")
    if len(pv_list) == 1:
      return single_pv_struct.wrap(value)
    else:
      result_list = []
      for p in pv_list:
        pv_data = Value(multi_pv_struct, {"pvName": p, "value": single_pv_struct.wrap(value)})
        result_list.append(pv_data)
      return Value(multi_response_struct, {"value": result_list})
    
print("Starting Archive Test Server!")
tester = ArchiveTester()
quickRPCServer(provider="Archive Test Server",
               prefix="",
               target=tester)