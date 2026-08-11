// Entry point for `dart run cera_ffi_flutter:install_web`.
//
// The same command as `dart run cera_ffi:install_web`, exposed here because a
// Flutter app depends on this package alone: `cera_ffi` is a transitive
// dependency, and a first-run setup step should not rest on whether `dart run`
// reaches executables in one.

import 'package:cera_ffi/install_web.dart';

Future<void> main(List<String> args) => installWeb(args);
