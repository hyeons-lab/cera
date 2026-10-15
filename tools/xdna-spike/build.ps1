# Builds spike.exe against the XRT runtime that ships inside the AMD NPU driver.
# Needs Visual Studio with the C++ workload, and the XRT headers in .\xrt (see README.md).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$vs = & 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe' -latest -products * -property installationPath
if (-not $vs) { throw "Visual Studio not found" }
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | % { if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] } }
# The DLL spike.exe loads at run time, so the import library matches it.
$dll = "$env:SystemRoot\System32\xrt_coreutil.dll"
if (-not (Test-Path $dll)) { throw "$dll not found; is the AMD NPU driver installed?" }
$names = dumpbin /nologo /exports $dll | % { if ($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)') { $matches[1] } }
if ($LASTEXITCODE -ne 0 -or $names.Count -eq 0) { throw "dumpbin found no exports in $dll" }
@("LIBRARY xrt_coreutil", "EXPORTS") + $names | Set-Content -Encoding ascii xrt_coreutil.def
lib /nologo /def:xrt_coreutil.def /machine:x64 /out:xrt_coreutil.lib | Out-Null
if ($LASTEXITCODE -ne 0) { throw "lib.exe failed to build xrt_coreutil.lib" }
$inc = "xrt\src\runtime_src\core\include"
if (-not (Test-Path "$inc\xrt\detail\version.h")) { throw "$inc\xrt\detail\version.h missing (see README.md)" }
cl /nologo /std:c++17 /Zc:__cplusplus /EHsc /O2 /MD /I $inc /I "$inc\xrt" spike.cpp xrt_coreutil.lib /Fe:spike.exe
"BUILD_EXIT=$LASTEXITCODE  exports=$($names.Count)"
exit $LASTEXITCODE
