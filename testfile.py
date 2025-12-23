from deltalake import DeltaTable
dt = DeltaTable("delta_demo_poc")
print(dt.to_pandas().shape[0])

