# Windows side of the `cera_ffi_flutter` FFI plugin.
#
# Mirrors linux/CMakeLists.txt: downloads the prebuilt cdylib the release
# pipeline publishes, verifies its SHA-256, and hands the DLL to Flutter's
# bundler so it lands next to the built executable, where
# `DynamicLibrary.open('cera_ffi.dll')` finds it.
cmake_minimum_required(VERSION 3.14)

set(PROJECT_NAME "cera_ffi_flutter")
project(${PROJECT_NAME} LANGUAGES CXX)

# Names a published release tag. Kept in lockstep with the Cargo workspace by
# scripts/bump-version.sh (which verifies the write landed); do not hand-edit.
set(CERA_VERSION "0.5.0")
set(CERA_LIB_LABEL "win32-x86-64")
set(CERA_LIB_FILE "cera_ffi.dll")

# x64 only: the release pipeline builds x86_64-pc-windows-msvc and nothing
# else. Windows on ARM has no published library.
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR
    "cera_ffi_flutter: only 64-bit Windows is supported.")
endif()

set(CERA_ARCHIVE "cera-ffi-${CERA_LIB_LABEL}-v${CERA_VERSION}.tar.gz")
set(CERA_URL
  "https://github.com/hyeons-lab/cera/releases/download/v${CERA_VERSION}/${CERA_ARCHIVE}")
set(CERA_DOWNLOAD_DIR "${CMAKE_CURRENT_BINARY_DIR}/cera-ffi")
set(CERA_LIB_PATH "${CERA_DOWNLOAD_DIR}/${CERA_LIB_FILE}")


# ── Local override ────────────────────────────────────────────────────────────
# A library sitting next to this CMakeLists always wins, mirroring the Apple
# podspec and SPM manifest. Two reasons it has to exist:
#
#  1. The download below targets a *published* release asset, so before the
#     first release that carries one there is nothing to fetch and the platform
#     cannot be built or tested at all.
#  2. It is the only way to try an unreleased engine build here.
#
# Populate it with:
#     cargo build -p cera-ffi --release --features ffi-buffer
#     cp target/release/cera_ffi.dll cera_ffi_flutter/windows/
#
# The `ffi-buffer` feature is not optional: without it the library loads and
# every Dart call then fails at `dlsym`. Gitignored, so it can only ever be
# something you put there deliberately.
set(CERA_LOCAL_LIB "${CMAKE_CURRENT_SOURCE_DIR}/cera_ffi.dll")

if(EXISTS "${CERA_LOCAL_LIB}")
  message(STATUS "cera_ffi_flutter: using locally-supplied ${CERA_LOCAL_LIB} (not verified)")
  set(CERA_LIB_PATH "${CERA_LOCAL_LIB}")
elseif(NOT EXISTS "${CERA_LIB_PATH}")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$")
    message(FATAL_ERROR
      "cera_ffi_flutter: no prebuilt cera-ffi library for Windows on ARM. "
      "Build cera-ffi from source and place ${CERA_LIB_FILE} in ${CMAKE_CURRENT_SOURCE_DIR}, "
      "or set CERA_FFI_LIB at runtime.")
  endif()

  file(MAKE_DIRECTORY "${CERA_DOWNLOAD_DIR}")

  message(STATUS "cera_ffi_flutter: fetching ${CERA_ARCHIVE}")
  file(DOWNLOAD "${CERA_URL}.sha256" "${CERA_DOWNLOAD_DIR}/${CERA_ARCHIVE}.sha256"
    STATUS CERA_SHA_STATUS
    TLS_VERIFY ON)
  list(GET CERA_SHA_STATUS 0 CERA_SHA_CODE)
  if(NOT CERA_SHA_CODE EQUAL 0)
    list(GET CERA_SHA_STATUS 1 CERA_SHA_ERROR)
    message(FATAL_ERROR "cera_ffi_flutter: could not download checksum: ${CERA_SHA_ERROR}")
  endif()

  file(READ "${CERA_DOWNLOAD_DIR}/${CERA_ARCHIVE}.sha256" CERA_SHA_LINE)
  string(REGEX MATCH "^[0-9a-fA-F]+" CERA_EXPECTED_SHA "${CERA_SHA_LINE}")
  if(NOT CERA_EXPECTED_SHA)
    message(FATAL_ERROR "cera_ffi_flutter: malformed checksum file for ${CERA_ARCHIVE}")
  endif()

  file(DOWNLOAD "${CERA_URL}" "${CERA_DOWNLOAD_DIR}/${CERA_ARCHIVE}"
    EXPECTED_HASH "SHA256=${CERA_EXPECTED_SHA}"
    STATUS CERA_DL_STATUS
    TLS_VERIFY ON
    SHOW_PROGRESS)
  list(GET CERA_DL_STATUS 0 CERA_DL_CODE)
  if(NOT CERA_DL_CODE EQUAL 0)
    list(GET CERA_DL_STATUS 1 CERA_DL_ERROR)
    message(FATAL_ERROR "cera_ffi_flutter: could not download ${CERA_ARCHIVE}: ${CERA_DL_ERROR}")
  endif()

  file(ARCHIVE_EXTRACT
    INPUT "${CERA_DOWNLOAD_DIR}/${CERA_ARCHIVE}"
    DESTINATION "${CERA_DOWNLOAD_DIR}")

  if(NOT EXISTS "${CERA_LIB_PATH}")
    message(FATAL_ERROR
      "cera_ffi_flutter: ${CERA_ARCHIVE} did not contain ${CERA_LIB_FILE}")
  endif()
endif()

set(cera_ffi_flutter_bundled_libraries
  "${CERA_LIB_PATH}"
  PARENT_SCOPE)
set(cera_ffi_flutter_plugins
  ""
  PARENT_SCOPE)
