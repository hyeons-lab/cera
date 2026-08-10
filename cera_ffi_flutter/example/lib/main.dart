// Example Flutter app for `cera_ffi_flutter`: pick a GGUF, chat with it, watch
// tokens stream in.
//
// The interesting part is that inference runs on a background isolate. Every
// `generate*` call blocks its thread for as long as decoding takes, so calling
// it on the UI isolate would freeze the frame pump. `Isolate.run` keeps the UI
// responsive and lets tokens arrive over a port.

import 'dart:async';
import 'dart:io' show Platform;
import 'dart:isolate';

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

void main() => runApp(const CeraExampleApp());

class CeraExampleApp extends StatelessWidget {
  const CeraExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cera Example',
      theme: ThemeData(
        colorSchemeSeed: Colors.indigo,
        useMaterial3: true,
      ),
      home: const ChatPage(),
    );
  }
}

/// One message in the transcript.
class Turn {
  Turn({required this.role, required this.text});

  final String role;
  String text;

  bool get isUser => role == 'user';
}

class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final _input = TextEditingController();
  final _scroll = ScrollController();
  final _turns = <Turn>[];

  String? _modelPath;
  String _status = 'No model loaded';
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    // Reading the backend report proves the native library resolved before the
    // user does anything, which makes a packaging problem obvious immediately
    // rather than at first generate.
    unawaited(_probeNativeLibrary());
  }

  @override
  void dispose() {
    _input.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _probeNativeLibrary() async {
    try {
      final report = cpuBackendReport();
      final version = ceraFfiVersion();
      setState(() => _status = 'cera $version · $report');
    } catch (err, stack) {
      // Log as well as display: the status line truncates, and the full
      // message is the only thing that identifies *which* resolution step
      // failed on a device.
      debugPrint('cera: native library failed to load: $err\n$stack');
      setState(() => _status = 'Native library failed to load: $err');
    }
  }

  Future<void> _pickModel() async {
    final result = await FilePicker.platform.pickFiles(
      dialogTitle: 'Choose a .gguf model',
      type: FileType.any,
    );
    final path = result?.files.single.path;
    if (path == null) return;
    setState(() {
      _modelPath = path;
      _status = 'Model: ${path.split(Platform.pathSeparator).last}';
      _turns.clear();
    });
  }

  Future<void> _send() async {
    final prompt = _input.text.trim();
    final modelPath = _modelPath;
    if (prompt.isEmpty || modelPath == null || _busy) return;

    _input.clear();
    setState(() {
      _busy = true;
      _turns.add(Turn(role: 'user', text: prompt));
      _turns.add(Turn(role: 'assistant', text: ''));
    });
    _scrollToBottom();

    final receive = ReceivePort();
    final sendPort = receive.sendPort;

    // Tokens stream back over the port; each one appends to the last turn.
    final sub = receive.listen((message) {
      if (message is String) {
        setState(() => _turns.last.text += message);
        _scrollToBottom();
      }
    });

    try {
      await Isolate.run(() => _generateOnIsolate(modelPath, prompt, sendPort));
    } catch (err) {
      setState(() => _turns.last.text = 'Error: $err');
    } finally {
      await sub.cancel();
      receive.close();
      setState(() => _busy = false);
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scroll.hasClients) return;
      _scroll.jumpTo(_scroll.position.maxScrollExtent);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cera'),
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(24),
          child: Padding(
            padding: const EdgeInsets.only(left: 16, right: 16, bottom: 8),
            child: Align(
              alignment: Alignment.centerLeft,
              child: Text(
                _status,
                style: Theme.of(context).textTheme.bodySmall,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ),
        ),
        actions: [
          IconButton(
            onPressed: _busy ? null : _pickModel,
            icon: const Icon(Icons.folder_open),
            tooltip: 'Open a .gguf model',
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: _turns.isEmpty
                ? const Center(child: Text('Open a .gguf model to start.'))
                : ListView.builder(
                    controller: _scroll,
                    padding: const EdgeInsets.all(12),
                    itemCount: _turns.length,
                    itemBuilder: (context, i) => _Bubble(turn: _turns[i]),
                  ),
          ),
          const Divider(height: 1),
          Padding(
            padding: const EdgeInsets.all(12),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _input,
                    enabled: _modelPath != null && !_busy,
                    onSubmitted: (_) => _send(),
                    decoration: const InputDecoration(
                      hintText: 'Ask something…',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                IconButton.filled(
                  onPressed: _modelPath != null && !_busy ? _send : null,
                  icon: _busy
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.send),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _Bubble extends StatelessWidget {
  const _Bubble({required this.turn});

  final Turn turn;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Align(
      alignment: turn.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        constraints: const BoxConstraints(maxWidth: 520),
        decoration: BoxDecoration(
          color: turn.isUser ? scheme.primaryContainer : scheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(14),
        ),
        child: Text(turn.text.isEmpty ? '…' : turn.text),
      ),
    );
  }
}

/// Runs on a background isolate: loads the model, renders the chat template,
/// generates, and posts decoded text back as it goes.
///
/// The engine and session are created here and dropped when the isolate exits.
/// A real app would keep them alive across turns; this reloads each time to
/// keep the example self-contained.
void _generateOnIsolate(String modelPath, String prompt, SendPort send) {
  final engine = CeraEngine.fromPath(
    modelPath,
    const EngineConfig(
      contextSize: 4096,
      // Auto probes the GPU backend (Metal on Apple) and falls back to CPU.
      backend: BackendPreference.auto,
      bundleRepo: null,
    ),
  );

  final rendered = engine.hasChatTemplate()
      ? engine.applyChatTemplate(
          [ChatMessage(role: 'user', content: prompt)],
          true,
        )
      : prompt;

  final session = engine.newSession(const SessionConfig(
    maxSeqLen: null,
    kvCompression: KvCompressionNone(),
    nKeep: 0,
    seed: null,
    ubatchSize: 512,
  ));

  session.appendTokens(engine.encodeTextSpecial(rendered, true));

  final out = session.generate(const GenerateOpts(
    maxTokens: 256,
    temperature: 0.7,
    topP: 0.95,
    topK: 40,
    minP: 0.0,
    repetitionPenalty: 1.1,
    stopTokens: <int>[],
    ignoreEos: false,
    grammar: null,
    grammarTriggerTokens: <int>[],
    flushEveryTokens: 0,
    flushEveryMs: 0,
  ));

  send.send(engine.decodeTokens(out.tokens));
}
