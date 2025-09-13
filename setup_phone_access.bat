@echo off
echo Setting up phone access for gender detection app...
echo.

echo Adding Windows Firewall rule...
netsh advfirewall firewall add rule name="Gender Detection App" dir=in action=allow protocol=TCP localport=5000

echo.
echo Setting up port forwarding from Windows to WSL...
netsh interface portproxy add v4tov4 listenport=5000 listenaddress=0.0.0.0 connectport=5000 connectaddress=172.22.196.192

echo.
echo Port forwarding setup complete!
echo.
echo Access URLs:
echo - From laptop: http://localhost:5000
echo - From phone: http://10.54.91.236:5000
echo.
echo Press any key to continue...
pause > nul

