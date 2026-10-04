$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$vs = & 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe' -latest -products * -property installationPath
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | % { if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] } }
$dll = (Get-ChildItem C:\Windows\System32\DriverStore\FileRepository -Directory -Filter "kipudrv.inf_amd64_*" | Sort LastWriteTime -Desc | Select -First 1).FullName + "\xrt_coreutil.dll"
if (-not (Test-Path $dll)) { throw "xrt_coreutil.dll not found" }
$names = dumpbin /nologo /exports $dll | % { if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)') { $matches[1] } }
@("LIBRARY xrt_coreutil", "EXPORTS") + $names | Set-Content -Encoding ascii xrt_coreutil.def
lib /nologo /def:xrt_coreutil.def /machine:x64 /out:xrt_coreutil.lib | Out-Null
$inc = "xrt\src\runtime_src\core\include"
cl /nologo /std:c++17 /Zc:__cplusplus /EHsc /O2 /MD /I $inc /I "$inc\xrt" spike.cpp xrt_coreutil.lib /Fe:spike.exe
"BUILD_EXIT=$LASTEXITCODE  exports=$($names.Count)"
