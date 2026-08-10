// Entry point for `dart run cera_ffi:install_web`.
//
// The implementation is in `lib/install_web.dart` so that
// `cera_ffi_flutter` can expose the same command without duplicating it.

import 'package:cera_ffi/install_web.dart';

Future<void> main(List<String> args) => installWeb(args);
