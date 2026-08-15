// Example Flutter app demonstrating `cera_ffi_flutter` across iOS, Android,
// macOS, Linux, Windows, and the Web.
//
// Demonstrates MVI (Model-View-Intent) architecture with clean model lifecycle
// management (guaranteed engine unloading and resource cleanup on model switch).
//
// Running locally:
//   flutter run -d macos
//   flutter run -d chrome

import 'dart:async';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'benchmark.dart';
import 'chat_controller.dart';
import 'chat_intent.dart';
import 'chat_state.dart';
import 'model_source.dart';
import 'widgets/bundle_picker_dialog.dart';
import 'widgets/message_composer.dart';
import 'widgets/message_list.dart';

void main() => runApp(const CeraExampleApp());

class CeraExampleApp extends StatelessWidget {
  const CeraExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    const bgDark = Color(0xFF0B0C0E);
    const surfaceDark = Color(0xFF14161B);
    const borderDark = Color(0xFF232732);
    const textPrimary = Color(0xFFF1F5F9);
    const textSecondary = Color(0xFF8E95A5);
    const accentBlue = Color(0xFF3B82F6);

    return MaterialApp(
      title: 'Cera',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: bgDark,
        canvasColor: bgDark,
        cardColor: surfaceDark,
        dividerColor: borderDark,
        colorScheme: const ColorScheme.dark(
          primary: accentBlue,
          surface: surfaceDark,
          outline: borderDark,
          outlineVariant: Color(0xFF1E222D),
          onSurface: textPrimary,
          onSurfaceVariant: textSecondary,
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: surfaceDark,
          elevation: 0,
          scrolledUnderElevation: 0,
          titleTextStyle: TextStyle(
            color: textPrimary,
            fontSize: 17,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      home: const ChatPage(),
    );
  }
}

/// Main chat interface built with MVI architecture.
class ChatPage extends StatefulWidget {
  const ChatPage({super.key});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  late final ChatController _controller = ChatController(
    defaultStoreDir: _storeDir,
  );
  final TextEditingController _inputController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _controller.addListener(_onStateChange);
    // Restore the previously loaded model if one was saved
    _controller.dispatch(const RestoreLastModelIntent());
  }

  void _onStateChange() {
    if (!mounted) return;
    if (_controller.value.isGenerating) {
      _scrollToBottom();
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 150),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  void dispose() {
    _controller.removeListener(_onStateChange);
    _controller.dispose();
    _inputController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  /// Supplies bundle cache directory for mobile platforms.
  Future<String?> _storeDir() async {
    if (kIsWeb) return null;
    final mobile =
        defaultTargetPlatform == TargetPlatform.android ||
        defaultTargetPlatform == TargetPlatform.iOS;
    if (!mobile) return null;
    return (await getApplicationSupportDirectory()).path;
  }

  /// Opens the bundle selector dialog and dispatches load intent.
  Future<void> _pickBundle(ChatState state) async {
    String? currentBundleName;
    String? currentQuant;
    final loaded = state.loadedModel;
    if (loaded is BundleModelSource) {
      currentBundleName = loaded.bundleName;
      currentQuant = loaded.quant;
    }

    final choice = await showDialog<BundleChoice>(
      context: context,
      builder: (_) => BundlePickerDialog(
        currentBundleName: currentBundleName,
        currentQuant: currentQuant,
      ),
    );

    if (choice == null || !mounted) return;

    _controller.dispatch(
      LoadBundleIntent(
        bundleName: choice.bundleName,
        quant: choice.quant,
        displayName: choice.displayName,
        storeDir: _storeDir,
      ),
    );
  }

  /// Opens a local .gguf file from device storage.
  Future<void> _pickLocalModel() async {
    try {
      final source = await pickModelSource(dialogTitle: 'Choose a .gguf model');
      if (source != null && mounted) {
        _controller.dispatch(LoadLocalModelIntent(source));
      }
    } catch (err) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not open local model: $err')),
        );
      }
    }
  }

  /// Attaches an image file for vision chat inference.
  Future<void> _pickImage() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.image,
        withData: true,
        dialogTitle: 'Select an image to attach',
      );
      final file = result?.files.single;
      if (file == null || !mounted) return;

      final bytes = file.bytes;
      if (bytes == null) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Could not read image file bytes')),
          );
        }
        return;
      }

      _controller.dispatch(AttachImageIntent(bytes: bytes, name: file.name));
    } catch (err) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Could not attach image: $err')));
      }
    }
  }

  void _sendMessage() {
    final prompt = _inputController.text.trim();
    if (prompt.isEmpty && _controller.value.pendingImageBytes == null) return;
    _inputController.clear();
    _controller.dispatch(SendMessageIntent(prompt));
  }

  void _showModelOptionsSheet(ChatState state) {
    showModalBottomSheet(
      context: context,
      backgroundColor: const Color(0xFF14161B),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (context) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                ListTile(
                  leading: const Icon(
                    Icons.cloud_download_outlined,
                    color: Color(0xFF60A5FA),
                  ),
                  title: const Text('Downloaded & Catalog Models'),
                  subtitle: const Text(
                    'Switch between cached models or download new ones',
                  ),
                  onTap: () {
                    Navigator.of(context).pop();
                    _pickBundle(state);
                  },
                ),
                ListTile(
                  leading: const Icon(
                    Icons.folder_open_outlined,
                    color: Color(0xFF34D399),
                  ),
                  title: const Text('Open Local .gguf File...'),
                  subtitle: const Text('Pick a model file from disk storage'),
                  onTap: () {
                    Navigator.of(context).pop();
                    _pickLocalModel();
                  },
                ),
                if (state.hasModel) ...[
                  const Divider(color: Color(0xFF232732)),
                  ListTile(
                    leading: const Icon(
                      Icons.eject_outlined,
                      color: Color(0xFFF87171),
                    ),
                    title: const Text(
                      'Unload Active Model',
                      style: TextStyle(color: Color(0xFFF87171)),
                    ),
                    subtitle: const Text(
                      'Release memory and close model engine',
                    ),
                    onTap: () {
                      Navigator.of(context).pop();
                      _controller.dispatch(const UnloadModelIntent());
                    },
                  ),
                ],
              ],
            ),
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<ChatState>(
      valueListenable: _controller,
      builder: (context, state, _) {
        return Scaffold(
          appBar: AppBar(
            title: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Cera'),
                const SizedBox(height: 2),
                Text(
                  state.status,
                  style: const TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.normal,
                    color: Color(0xFF8E95A5),
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
            bottom: PreferredSize(
              preferredSize: Size.fromHeight(state.isLoading ? 3 : 1),
              child: state.isLoading
                  ? LinearProgressIndicator(
                      value: state.downloadFraction,
                      minHeight: 3,
                      backgroundColor: const Color(0xFF1E222D),
                      valueColor: const AlwaysStoppedAnimation<Color>(
                        Color(0xFF3B82F6),
                      ),
                    )
                  : Container(color: const Color(0xFF1E222D), height: 1),
            ),
            actions: [
              IconButton(
                icon: const Icon(Icons.delete_outline_rounded),
                tooltip: 'Clear transcript',
                onPressed: state.turns.isEmpty
                    ? null
                    : () => _controller.dispatch(const ClearTranscriptIntent()),
              ),
              IconButton(
                icon: const Icon(Icons.speed),
                tooltip: state.hasModel
                    ? 'CPU vs GPU Benchmark'
                    : 'Load a model first to benchmark',
                onPressed: state.hasModel
                    ? () {
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (_) =>
                                BenchmarkPage(model: state.loadedModel!),
                          ),
                        );
                      }
                    : null,
              ),
              IconButton(
                icon: const Icon(Icons.tune_rounded),
                tooltip: 'Model Options',
                onPressed: () => _showModelOptionsSheet(state),
              ),
            ],
          ),
          body: Column(
            children: [
              Expanded(
                child: MessageList(
                  turns: state.turns,
                  scrollController: _scrollController,
                ),
              ),
              MessageComposer(
                controller: _inputController,
                isBusy: state.isBusy,
                isGenerating: state.isGenerating,
                canAttachImage: state.canAttachImage,
                pendingImageBytes: state.pendingImageBytes,
                pendingImageName: state.pendingImageName,
                onSend: _sendMessage,
                onStop: () =>
                    _controller.dispatch(const StopGenerationIntent()),
                onPickImage: _pickImage,
                onClearImage: () =>
                    _controller.dispatch(const ClearAttachedImageIntent()),
              ),
            ],
          ),
        );
      },
    );
  }
}
