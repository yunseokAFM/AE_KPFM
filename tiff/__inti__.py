from SmartRemote import SmartRemote as SR


# Initate SmartRemote Object
sr = SR()

# Check if SmartRemote is connected
connection = sr.check_connection()
print(f'SmartRemote is {connection}')


# Check if PowerScript is running
result = sr.check_status
print(f'PowerScript running is {connection}')

# Run PowerScript and wait until PowerScript running is done
script = "spm.zstage.move(10,0.8);"
result = sr.run(script)
print(result)

# Query X,Y,Z stage position
result = sr.query_stage_pos()
print(result)


# query scan status
reply = sr.query_scan_status()
print(reply)

# query geometry
reply = sr.query_geometry()
print(reply)