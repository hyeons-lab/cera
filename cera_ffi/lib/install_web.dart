/// Installs the web runtime into an app's `web/` directory.
///
/// The implementation behind `dart run cera_ffi:install_web` and
/// `dart run cera_ffi_flutter:install_web`. It is a library rather than a
/// script body so that both packages can expose the command: a Flutter app
/// depends only on `cera_ffi_flutter`, and whether `dart run` reaches an
/// executable in a transitive dependency is not something to make a
/// getting-started step depend on.
library;

// Three files have to be served for the web implementation to work:
//
//   cera_worker.js     the RPC host, shipped inside this package
//   cera_wasm.js       wasm-bindgen's loader
//   cera_wasm_bg.wasm  the engine
//
// Only the worker ships in the pub archive. The other two are build outputs of
// the Rust crate, and this package follows the same rule as every other target
// here: binaries come from a release, never from the archive or from git. That
// is also why this is a tool rather than a `flutter: assets:` entry, and it
// matches how the rest of the Flutter web ecosystem ships wasm (`sqlite3.wasm`,
// `drift_worker.js`): the app owns its `web/` directory.
//
// The two names are not free choices: wasm-bindgen's `--target web` loader
// resolves its `.wasm` sibling as `new URL('cera_wasm_bg.wasm', import.meta.url)`,
// so the pair must keep those names and stay in one directory. Release assets
// are versioned, so this renames on write.
//
// Known gap: the download is not checksummed, unlike the Linux and Windows
// libraries the CMake wiring fetches. Those verify against a hash committed
// next to the wiring; there is no equivalent published for the web artifacts
// yet. Until there is, `--from` against a local build is the verifiable path.

import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data' show BytesBuilder;

const _defaultOut = 'web/cera';

Future<void> installWeb(List<String> args) async {
  if (args.contains('-h') || args.contains('--help')) {
    stdout.writeln(_usage);
    return;
  }
  final from = _valueOf(args, '--from');
  final out = Directory(_valueOf(args, '--out') ?? _defaultOut);
  final force = args.contains('--force');

  final pubspec = await _readPubspec();
  await out.create(recursive: true);

  await _installWorker(out, force: force);
  if (from != null) {
    await _copyModule(Directory(from), out, force: force);
  } else {
    await _downloadModule(pubspec, out, force: force);
  }

  stdout
    ..writeln()
    ..writeln('Installed into ${out.path}/')
    ..writeln()
    ..writeln('Point the engine at it if you changed --out:')
    ..writeln(
      '  Cera.openBytes(bytes, options: CeraOptions(web: CeraWebAssets(',
    )
    ..writeln("    workerUrl: '${_urlOf(out)}/cera_worker.js',")
    ..writeln("    moduleUrl: '${_urlOf(out)}/cera_wasm.js',")
    ..writeln('  )));');
}

const _usage = '''
Install the Cera web runtime into an app's web/ directory.

Usage: dart run cera_ffi:install_web [options]
       dart run cera_ffi_flutter:install_web [options]   (from a Flutter app)

  --from DIR   Copy cera_wasm.js and cera_wasm_bg.wasm from a local wasm-pack
               build instead of downloading them. Use the output of
               `just wasm-web-wgpu` (cera-wasm/examples/webgpu/pkg).
  --out DIR    Where to install. Default: $_defaultOut
  --force      Overwrite files that are already there.
  -h, --help   Show this message.
''';

String? _valueOf(List<String> args, String flag) {
  final i = args.indexOf(flag);
  if (i < 0) return null;
  if (i + 1 >= args.length) {
    throw ArgumentError('$flag needs a value');
  }
  return args[i + 1];
}

/// The `web/`-relative URL an app will serve [dir] at.
///
/// Flutter serves the contents of `web/` at the site root, so the leading
/// `web/` is exactly the part that does not appear in the URL.
String _urlOf(Directory dir) {
  final raw = dir.path.replaceAll(r'\', '/');
  // An absolute path is returned untouched: there is no way to know what URL a
  // server maps it to, and guessing produces a confidently wrong hint.
  if (raw.startsWith('/')) return raw;
  // Normalized so `./web/cera` and `web/./cera` reduce to `web/cera`.
  final path = _normalizePath(raw);
  return path.startsWith('web/') ? path.substring(4) : path;
}

/// Collapses `.` segments and duplicate slashes in a relative path.
///
/// Hand-rolled rather than `package:path`, which this package does not depend
/// on and would not gain anything else from.
String _normalizePath(String path) {
  final parts = path.split('/').where((s) => s.isNotEmpty && s != '.').toList();
  return parts.join('/');
}

/// This package's own `version:` and `repository:`, read from the pubspec next
/// to the resolved `lib/` directory.
///
/// Parsed with a regex rather than a YAML dependency: two scalar fields at the
/// top level of a file this package controls do not justify one, and a tool
/// that pulls in dependencies is a tool that can fail to resolve.
Future<({String version, String repository})> _readPubspec() async {
  final lib = await Isolate.resolvePackageUri(
    Uri.parse('package:cera_ffi/cera_ffi.dart'),
  );
  if (lib == null) {
    throw StateError(
      'could not resolve package:cera_ffi; run this via `dart run`',
    );
  }
  final file = File.fromUri(lib.resolve('../pubspec.yaml'));
  final text = await file.readAsString();
  String field(String name) {
    final match = RegExp(
      '^$name:\\s*(\\S+)\\s*\$',
      multiLine: true,
    ).firstMatch(text);
    if (match == null) {
      throw StateError('no `$name:` in ${file.path}');
    }
    return match.group(1)!;
  }

  return (version: field('version'), repository: field('repository'));
}

Future<void> _installWorker(Directory out, {required bool force}) async {
  final uri = await Isolate.resolvePackageUri(
    Uri.parse('package:cera_ffi/src/web/cera_worker.js'),
  );
  if (uri == null) {
    throw StateError('could not resolve the bundled cera_worker.js');
  }
  await _write(
    out,
    'cera_worker.js',
    await File.fromUri(uri).readAsBytes(),
    force: force,
  );
}

Future<void> _copyModule(
  Directory from,
  Directory out, {
  required bool force,
}) async {
  for (final name in const ['cera_wasm.js', 'cera_wasm_bg.wasm']) {
    final source = File('${from.path}/$name');
    if (!source.existsSync()) {
      throw StateError(
        'no $name in ${from.path}. Build it first:\n'
        '  just wasm-web-wgpu   (writes cera-wasm/examples/webgpu/pkg)',
      );
    }
    await _write(out, name, await source.readAsBytes(), force: force);
  }
}

Future<void> _downloadModule(
  ({String version, String repository}) pubspec,
  Directory out, {
  required bool force,
}) async {
  final base = '${pubspec.repository}/releases/download/v${pubspec.version}';
  const assets = {
    'cera_wasm.js': 'cera-wasm-web.js',
    'cera_wasm_bg.wasm': 'cera-wasm-web_bg.wasm',
  };
  final client = HttpClient();
  try {
    for (final entry in assets.entries) {
      // Checked before the request, not after the download: re-running the
      // tool should not pull ~3 MB of wasm just to discard it.
      if (File('${out.path}/${entry.key}').existsSync() && !force) {
        stdout.writeln(
          '  skip  ${entry.key} (already present; --force to overwrite)',
        );
        continue;
      }
      final url = '$base/${entry.value}';
      stdout.writeln('Fetching $url');
      final request = await client.getUrl(Uri.parse(url));
      final response = await request.close();
      if (response.statusCode != 200) {
        throw StateError(
          'GET $url returned ${response.statusCode}. The release for '
          'v${pubspec.version} may not carry the web artifacts; build them '
          'locally instead:\n'
          '  just wasm-web-wgpu\n'
          '  dart run cera_ffi:install_web --from cera-wasm/examples/webgpu/pkg',
        );
      }
      // A BytesBuilder, not a growing `List<int>`: the wasm is ~3 MB and a
      // boxed int list costs several times that plus repeated regrowth.
      final builder = BytesBuilder(copy: false);
      await response.forEach(builder.add);
      await _write(out, entry.key, builder.takeBytes(), force: force);
    }
  } finally {
    client.close();
  }
}

Future<void> _write(
  Directory out,
  String name,
  List<int> bytes, {
  required bool force,
}) async {
  final file = File('${out.path}/$name');
  if (file.existsSync() && !force) {
    stdout.writeln('  skip  $name (already present; --force to overwrite)');
    return;
  }
  await file.writeAsBytes(bytes);
  stdout.writeln('  write $name (${_size(bytes.length)})');
}

String _size(int bytes) =>
    bytes < 1024 * 1024
        ? '${(bytes / 1024).toStringAsFixed(0)} KB'
        : '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
