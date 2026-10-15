// Example Flutter app demonstrating `cera_ffi_flutter` across iOS, Android,
// macOS, Linux, Windows, and the Web.
//
// Demonstrates MVI (Model-View-Intent) architecture with clean model lifecycle
// management (guaranteed engine unloading and resource cleanup on model switch).
//
// Running locally:
//   flutter run -d macos
//   flutter run -d chrome

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart' hide ModelSource;

import 'app_constants.dart';
import 'chat_controller.dart';
import 'chat_intent.dart';
import 'chat_state.dart';
import 'model_source.dart';
import 'widgets/bundle_picker_dialog.dart';
import 'widgets/message_composer.dart';
import 'widgets/message_list.dart';
import 'widgets/tts_studio_view.dart';
import 'widgets/voice_persona_picker.dart';

void main() => runApp(const CeraExampleApp());

class CeraExampleApp extends StatelessWidget {
  const CeraExampleApp({super.key});

  static const _kBgDark = Color(0xFF08090C);
  static const _kSurfaceDark = Color(0xFF11141C);
  static const _kBorderDark = Color(0xFF222736);
  static const _kOutlineVariantDark = Color(0xFF1A1E2B);
  static const _kTextPrimary = Color(0xFFF1F5F9);
  static const _kTextSecondary = Color(0xFF94A3B8);
  static const _kAccentBlue = Color(0xFF3B82F6);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: AppStrings.appTitle,
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: _kBgDark,
        canvasColor: _kBgDark,
        cardColor: _kSurfaceDark,
        dividerColor: _kBorderDark,
        colorScheme: const ColorScheme.dark(
          primary: _kAccentBlue,
          surface: _kSurfaceDark,
          outline: _kBorderDark,
          outlineVariant: _kOutlineVariantDark,
          onSurface: _kTextPrimary,
          onSurfaceVariant: _kTextSecondary,
        ),
        appBarTheme: const AppBarTheme(
          backgroundColor: _kSurfaceDark,
          elevation: 0,
          scrolledUnderElevation: 0,
          titleTextStyle: TextStyle(
            color: _kTextPrimary,
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
  static const _kPickModelDialogTitle = 'Choose a .gguf model';
  static const _kPickImageDialogTitle = 'Select an image to attach';
  static const _kUnreadableImageBytesError = 'Could not read image file bytes';

  // AppBar & Action labels
  static const _kChatSegmentLabel = 'Chat';
  static const _kTtsStudioSegmentLabel = 'TTS Studio';
  static const _kSwitchModelTooltip = 'Select or download model';
  static const _kSelectModelLabel = 'Select Model';
  static const _kClearTranscriptTooltip = 'Clear transcript';
  static const _kSettingsTooltip = 'Settings & Models';

  // Settings sheet section headers & copy
  static const _kSettingsTitle = 'Settings & Models';
  static const _kSectionModels = 'MODELS';
  static const _kCatalogModelsTitle = 'Downloaded & Catalog Models';
  static const _kCatalogModelsSubtitle =
      'Switch between cached models or download new ones';
  static const _kOpenLocalTitle = 'Open Local .gguf File...';
  static const _kOpenLocalSubtitle = 'Pick a model file from disk storage';
  static const _kUnloadModelTitle = 'Unload Active Model';
  static const _kUnloadModelSubtitle = 'Release memory and close model engine';

  static const _kSectionSampling = 'SAMPLING & GENERATION';
  static const _kSamplingTemperatureTitle = 'Sampling Temperature';
  static const _kSamplingAutoHint =
      'Auto uses model defaults (0.0 greedy argmax on WebGPU text, 0.7 on audio).';
  static const _kSamplingGreedyHint =
      'Greedy (0.0): deterministic argmax decoding with GPU fast path.';
  static const _kSamplingStochasticHint =
      'Stochastic (> 0.0): nucleus / top-k sampling for varied responses.';
  static const _kPresetGreedyLabel = 'Greedy (0.0)';
  static const _kMaxOutputTokensTitle = 'Max Output Tokens';
  static const _kMaxOutputTokensSubtitle =
      'Maximum number of tokens generated per assistant turn.';

  static const _kSectionInference = 'INFERENCE & OPTIMIZATION';
  static const _kComputeBackendTitle = 'Compute Backend';
  static const _kComputeBackendSubtitle =
      'Choose between Auto (GPU / NPU / Fallback), GPU (WebGPU / Metal), NPU (Qualcomm Hexagon), or CPU.';
  static const _kBackendAutoLabel = 'Auto (GPU / Fallback)';
  static const _kBackendGpuLabel = 'GPU (WebGPU / Metal)';
  static const _kBackendHexagonLabel = 'NPU (Qualcomm Hexagon)';
  static const _kBackendCpuLabel = 'CPU (WASM / CPU)';
  static const _kTurboQuantTitle = 'TurboQuant KV Compression';
  static const _kTurboQuantSubtitle =
      'Compresses KV cache to 3-bit keys / 2-bit values for lower memory footprint and faster multi-turn attention. (Default: Off)';
  static const _kVisionMaxDimTitle = 'Vision Max Image Dimension';
  static const _kVisionMaxDimSubtitle =
      'Caps the long side of input images before ViT patch encoding to minimize prompt latency.';
  static const _kImageDimFastLabel = '256 px (Fast)';
  static const _kImageDimHdLabel = '512 px (HD)';
  static const _kImageDimNativeLabel = 'Native / Off';

  static const _kVoiceModeTitle = 'Voice Mode';
  static const _kVoiceModeBarLabel = 'Voice Mode:';
  static const _kVoiceModeSubtitle =
      'Select how audio-capable models interact with speech (Speech to Text ASR, Interleaved Voice Chat, Text to Speech TTS, or Text Only).';
  static const _kVoiceModeAsrLabel = 'Speech to Text (ASR)';
  static const _kVoiceModeInterleavedLabel = 'Voice Chat (Interleaved)';
  static const _kVoiceModeChatShortLabel = 'Voice Chat';
  static const _kVoiceModeTtsLabel = 'Text to Speech (TTS)';
  static const _kVoiceModeTextOnlyMenuLabel = 'Text Only (No Audio)';
  static const _kVoiceModeTextOnlyChipLabel = 'Text Only';

  static const _kSectionBackendDevice = 'BACKEND & DEVICE';

  late final ChatController _controller = ChatController(
    defaultStoreDir: _storeDir,
  );
  final TextEditingController _inputController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  bool _scrollPending = false;

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
      if (!_scrollController.hasClients) return;
      final pos = _scrollController.position;
      final distanceToBottom = pos.maxScrollExtent - pos.pixels;
      // Only auto-scroll if the user is already near the bottom (within 120px)
      if (distanceToBottom < 120) {
        _scrollToBottom(isStreaming: true);
      }
    }
  }

  void _scrollToBottom({bool isStreaming = false}) {
    if (_scrollPending) return;
    _scrollPending = true;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _scrollPending = false;
      if (!mounted || !_scrollController.hasClients) return;
      final target = _scrollController.position.maxScrollExtent;
      if (isStreaming) {
        _scrollController.jumpTo(target);
      } else {
        _scrollController.animateTo(
          target,
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
    final loaded = state.loadedModel;
    final bundleModel = loaded is BundleModelSource ? loaded : null;

    final choice = await showDialog<BundleChoice>(
      context: context,
      builder: (_) => BundlePickerDialog(
        currentBundleName: bundleModel?.bundleName,
        currentQuant: bundleModel?.quant,
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
      final source = await pickModelSource(dialogTitle: _kPickModelDialogTitle);
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
        dialogTitle: _kPickImageDialogTitle,
      );
      final file = result?.files.single;
      if (file == null || !mounted) return;

      final bytes = file.bytes;
      if (bytes == null) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text(_kUnreadableImageBytesError)),
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
    _scrollToBottom();
    _controller.dispatch(SendMessageIntent(prompt));
  }

  void _onSelectStarterPrompt(String prompt) {
    _inputController.text = prompt;
    _sendMessage();
  }

  static String _samplingHintFor(double? temp) {
    if (temp == null) return _kSamplingAutoHint;
    if (temp <= 0.0) return _kSamplingGreedyHint;
    return _kSamplingStochasticHint;
  }

  void _showSettingsSheet(ChatState state) {
    final theme = Theme.of(context);
    showModalBottomSheet(
      context: context,
      backgroundColor: theme.colorScheme.surface,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (context) {
        return ListenableBuilder(
          listenable: _controller,
          builder: (context, _) {
            final currentState = _controller.value;
            final currentSettings = currentState.settings;
            final activeTemp = currentSettings.temperature;
            final sliderValue = (activeTemp ?? ChatDefaults.greedyTemperature)
                .clamp(
                  ChatDefaults.greedyTemperature,
                  ChatDefaults.maxSliderTemperature,
                );

            return SafeArea(
              child: Padding(
                padding: const EdgeInsets.symmetric(
                  vertical: 12,
                  horizontal: 16,
                ),
                child: SingleChildScrollView(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Center(
                        child: Container(
                          width: 36,
                          height: 4,
                          margin: const EdgeInsets.only(bottom: 16),
                          decoration: BoxDecoration(
                            color: theme.colorScheme.outlineVariant,
                            borderRadius: BorderRadius.circular(2),
                          ),
                        ),
                      ),
                      Row(
                        children: [
                          Icon(
                            Icons.settings_outlined,
                            size: 20,
                            color: theme.colorScheme.primary,
                          ),
                          const SizedBox(width: 8),
                          Text(
                            _kSettingsTitle,
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                              color: theme.colorScheme.onSurface,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      Text(
                        _kSectionModels,
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0.8,
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                      const SizedBox(height: 6),
                      ListTile(
                        contentPadding: EdgeInsets.zero,
                        leading: Icon(
                          Icons.cloud_download_outlined,
                          color: theme.colorScheme.primary,
                        ),
                        title: const Text(_kCatalogModelsTitle),
                        subtitle: const Text(_kCatalogModelsSubtitle),
                        onTap: () {
                          Navigator.of(context).pop();
                          _pickBundle(_controller.value);
                        },
                      ),
                      ListTile(
                        contentPadding: EdgeInsets.zero,
                        leading: Icon(
                          Icons.folder_open_outlined,
                          color: theme.colorScheme.primary,
                        ),
                        title: const Text(_kOpenLocalTitle),
                        subtitle: const Text(_kOpenLocalSubtitle),
                        onTap: () {
                          Navigator.of(context).pop();
                          _pickLocalModel();
                        },
                      ),
                      if (currentState.hasModel) ...[
                        ListTile(
                          contentPadding: EdgeInsets.zero,
                          leading: Icon(
                            Icons.eject_outlined,
                            color: theme.colorScheme.error,
                          ),
                          title: Text(
                            _kUnloadModelTitle,
                            style: TextStyle(color: theme.colorScheme.error),
                          ),
                          subtitle: const Text(_kUnloadModelSubtitle),
                          onTap: () {
                            Navigator.of(context).pop();
                            _controller.dispatch(const UnloadModelIntent());
                          },
                        ),
                      ],
                      Divider(color: theme.dividerColor, height: 24),
                      Text(
                        _kSectionSampling,
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0.8,
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            _kSamplingTemperatureTitle,
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w500,
                              color: theme.colorScheme.onSurface,
                            ),
                          ),
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 3,
                            ),
                            decoration: BoxDecoration(
                              color: theme.colorScheme.primary.withValues(
                                alpha: 0.14,
                              ),
                              borderRadius: BorderRadius.circular(6),
                              border: Border.all(
                                color: theme.colorScheme.primary.withValues(
                                  alpha: 0.3,
                                ),
                              ),
                            ),
                            child: Text(
                              currentSettings.samplingLabel,
                              style: TextStyle(
                                fontSize: 11.5,
                                fontWeight: FontWeight.w700,
                                color: theme.colorScheme.primary,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Text(
                        _samplingHintFor(activeTemp),
                        style: TextStyle(
                          fontSize: 12,
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                      const SizedBox(height: 10),
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: [
                          ChoiceChip(
                            label: const Text(
                              AppStrings.samplingAuto,
                              style: TextStyle(fontSize: 11.5),
                            ),
                            selected: activeTemp == null,
                            onSelected: (_) => _controller.dispatch(
                              const UpdateSettingsIntent(
                                clearTemperature: true,
                              ),
                            ),
                          ),
                          ChoiceChip(
                            label: const Text(
                              _kPresetGreedyLabel,
                              style: TextStyle(fontSize: 11.5),
                            ),
                            selected:
                                activeTemp != null &&
                                (activeTemp - ChatDefaults.greedyTemperature)
                                        .abs() <
                                    0.01,
                            onSelected: (_) => _controller.dispatch(
                              const UpdateSettingsIntent(
                                temperature: ChatDefaults.greedyTemperature,
                              ),
                            ),
                          ),
                          ChoiceChip(
                            label: const Text(
                              AppStrings.samplingBalanced,
                              style: TextStyle(fontSize: 11.5),
                            ),
                            selected:
                                activeTemp != null &&
                                (activeTemp - ChatDefaults.balancedTemperature)
                                        .abs() <
                                    0.01,
                            onSelected: (_) => _controller.dispatch(
                              const UpdateSettingsIntent(
                                temperature: ChatDefaults.balancedTemperature,
                              ),
                            ),
                          ),
                          ChoiceChip(
                            label: const Text(
                              AppStrings.samplingCreative,
                              style: TextStyle(fontSize: 11.5),
                            ),
                            selected:
                                activeTemp != null &&
                                (activeTemp - ChatDefaults.creativeTemperature)
                                        .abs() <
                                    0.01,
                            onSelected: (_) => _controller.dispatch(
                              const UpdateSettingsIntent(
                                temperature: ChatDefaults.creativeTemperature,
                              ),
                            ),
                          ),
                        ],
                      ),
                      Slider(
                        value: sliderValue,
                        min: ChatDefaults.greedyTemperature,
                        max: ChatDefaults.maxSliderTemperature,
                        divisions: 30,
                        label: sliderValue.toStringAsFixed(2),
                        onChanged: (val) {
                          final rounded = (val * 20).round() / 20.0;
                          _controller.dispatch(
                            UpdateSettingsIntent(temperature: rounded),
                          );
                        },
                      ),
                      ListTile(
                        contentPadding: EdgeInsets.zero,
                        title: const Text(_kMaxOutputTokensTitle),
                        subtitle: Text(
                          _kMaxOutputTokensSubtitle,
                          style: TextStyle(
                            fontSize: 12,
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                        trailing: DropdownButton<int>(
                          value:
                              ChatDefaults.maxTokenChoices.contains(
                                currentSettings.maxTokens,
                              )
                              ? currentSettings.maxTokens
                              : ChatDefaults.maxTokens,
                          dropdownColor: theme.colorScheme.surface,
                          underline: const SizedBox.shrink(),
                          items: [
                            for (final count in ChatDefaults.maxTokenChoices)
                              DropdownMenuItem(
                                value: count,
                                child: Text('$count tokens'),
                              ),
                          ],
                          onChanged: (val) {
                            if (val == null) return;
                            _controller.dispatch(
                              UpdateSettingsIntent(maxTokens: val),
                            );
                          },
                        ),
                      ),
                      Divider(color: theme.dividerColor, height: 24),
                      Text(
                        _kSectionInference,
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0.8,
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                      const SizedBox(height: 8),
                      ListTile(
                        contentPadding: EdgeInsets.zero,
                        title: const Text(_kComputeBackendTitle),
                        subtitle: Text(
                          _kComputeBackendSubtitle,
                          style: TextStyle(
                            fontSize: 12,
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                        trailing: DropdownButton<CeraBackend>(
                          value: currentSettings.backend,
                          dropdownColor: theme.colorScheme.surface,
                          underline: const SizedBox.shrink(),
                          items: const [
                            DropdownMenuItem(
                              value: CeraBackend.auto,
                              child: Text(_kBackendAutoLabel),
                            ),
                            DropdownMenuItem(
                              value: CeraBackend.gpu,
                              child: Text(_kBackendGpuLabel),
                            ),
                            DropdownMenuItem(
                              value: CeraBackend.hexagon,
                              child: Text(_kBackendHexagonLabel),
                            ),
                            DropdownMenuItem(
                              value: CeraBackend.cpu,
                              child: Text(_kBackendCpuLabel),
                            ),
                          ],
                          onChanged: (val) {
                            if (val == null) return;
                            _controller.dispatch(
                              UpdateSettingsIntent(backend: val),
                            );
                          },
                        ),
                      ),
                      const SizedBox(height: 8),
                      SwitchListTile(
                        contentPadding: EdgeInsets.zero,
                        title: const Text(_kTurboQuantTitle),
                        subtitle: Text(
                          _kTurboQuantSubtitle,
                          style: TextStyle(
                            fontSize: 12,
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                        value: currentSettings.turboQuant,
                        onChanged: (val) {
                          _controller.dispatch(
                            UpdateSettingsIntent(turboQuant: val),
                          );
                        },
                      ),
                      const SizedBox(height: 8),
                      ListTile(
                        contentPadding: EdgeInsets.zero,
                        title: const Text(_kVisionMaxDimTitle),
                        subtitle: Text(
                          _kVisionMaxDimSubtitle,
                          style: TextStyle(
                            fontSize: 12,
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                        trailing: DropdownButton<int?>(
                          value: currentSettings.maxImageLongSize,
                          dropdownColor: theme.colorScheme.surface,
                          underline: const SizedBox.shrink(),
                          items: const [
                            DropdownMenuItem(
                              value: ChatDefaults.maxImageLongSize,
                              child: Text(_kImageDimFastLabel),
                            ),
                            DropdownMenuItem(
                              value: ChatDefaults.hdImageLongSize,
                              child: Text(_kImageDimHdLabel),
                            ),
                            DropdownMenuItem(
                              value: null,
                              child: Text(_kImageDimNativeLabel),
                            ),
                          ],
                          onChanged: (val) {
                            _controller.dispatch(
                              UpdateSettingsIntent(
                                maxImageLongSize: val,
                                clearMaxImageLongSize: val == null,
                              ),
                            );
                          },
                        ),
                      ),
                      const SizedBox(height: 8),
                      ListTile(
                        contentPadding: EdgeInsets.zero,
                        title: const Text(_kVoiceModeTitle),
                        subtitle: Text(
                          _kVoiceModeSubtitle,
                          style: TextStyle(
                            fontSize: 12,
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                        trailing: DropdownButton<AudioChatMode>(
                          value: currentSettings.audioChatMode,
                          dropdownColor: theme.colorScheme.surface,
                          underline: const SizedBox.shrink(),
                          items: const [
                            DropdownMenuItem(
                              value: AudioChatMode.speechToText,
                              child: Text(_kVoiceModeAsrLabel),
                            ),
                            DropdownMenuItem(
                              value: AudioChatMode.interleaved,
                              child: Text(_kVoiceModeInterleavedLabel),
                            ),
                            DropdownMenuItem(
                              value: AudioChatMode.textToSpeech,
                              child: Text(_kVoiceModeTtsLabel),
                            ),
                            DropdownMenuItem(
                              value: AudioChatMode.textOnly,
                              child: Text(_kVoiceModeTextOnlyMenuLabel),
                            ),
                          ],
                          onChanged: (val) {
                            if (val == null) return;
                            _controller.dispatch(
                              UpdateSettingsIntent(audioChatMode: val),
                            );
                          },
                        ),
                      ),
                      if ((currentState.audioProfile?.hasVoices ?? false) &&
                          (currentSettings.audioChatMode ==
                                  AudioChatMode.interleaved ||
                              currentSettings.audioChatMode ==
                                  AudioChatMode.textToSpeech)) ...[
                        const SizedBox(height: 8),
                        VoicePersonaPicker(
                          profile: currentState.audioProfile,
                          saved: currentSettings.chatVoice,
                          onChanged: (val) {
                            _controller.dispatch(
                              UpdateSettingsIntent(
                                chatVoice: val,
                                clearChatVoice: val == null,
                              ),
                            );
                          },
                        ),
                      ],
                      if (currentState.hasModel) ...[
                        Divider(color: theme.dividerColor, height: 24),
                        Text(
                          _kSectionBackendDevice,
                          style: TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                            letterSpacing: 0.8,
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: theme.colorScheme.outlineVariant.withValues(
                              alpha: 0.25,
                            ),
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(
                              color: theme.colorScheme.outlineVariant,
                            ),
                          ),
                          child: Row(
                            children: [
                              Icon(
                                Icons.developer_board_rounded,
                                size: 24,
                                color: theme.colorScheme.primary,
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      currentState.backend ??
                                          AppStrings.unknownBackend,
                                      style: const TextStyle(
                                        fontSize: 13,
                                        fontWeight: FontWeight.w500,
                                      ),
                                    ),
                                    const SizedBox(height: 2),
                                    Text(
                                      'Active model: ${currentState.loadedModel?.name ?? ""}',
                                      style: TextStyle(
                                        fontSize: 11,
                                        color:
                                            theme.colorScheme.onSurfaceVariant,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                      const SizedBox(height: 12),
                    ],
                  ),
                ),
              ),
            );
          },
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return ValueListenableBuilder<ChatState>(
      valueListenable: _controller,
      builder: (context, state, _) {
        final hasAudioIn = state.capabilities?.audioIn ?? false;
        final hasAudioOut = state.capabilities?.audioOut ?? false;
        final loadedModelName = state.loadedModel?.name;

        return Scaffold(
          appBar: AppBar(
            title: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text(AppStrings.appTitle),
                    const SizedBox(width: 8),
                    Flexible(
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 6,
                          vertical: 2,
                        ),
                        decoration: BoxDecoration(
                          color: theme.colorScheme.primaryContainer,
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text(
                          AppStrings.appVersion,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.bold,
                            color: theme.colorScheme.onPrimaryContainer,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 6),
                    InkWell(
                      onTap: () => _showSettingsSheet(state),
                      borderRadius: BorderRadius.circular(6),
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 6,
                          vertical: 2,
                        ),
                        decoration: BoxDecoration(
                          color: theme.colorScheme.primary.withValues(
                            alpha: 0.14,
                          ),
                          borderRadius: BorderRadius.circular(6),
                          border: Border.all(
                            color: theme.colorScheme.primary.withValues(
                              alpha: 0.3,
                            ),
                          ),
                        ),
                        child: Text(
                          state.settings.samplingLabel,
                          style: TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.w600,
                            color: theme.colorScheme.primary,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 2),
                Text(
                  state.status,
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.normal,
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
            bottom: PreferredSize(
              preferredSize: const Size.fromHeight(3),
              child: state.isLoading
                  ? LinearProgressIndicator(
                      value: state.downloadFraction,
                      minHeight: 3,
                      backgroundColor: theme.colorScheme.outlineVariant,
                      valueColor: AlwaysStoppedAnimation<Color>(
                        theme.colorScheme.primary,
                      ),
                    )
                  : Container(
                      color: theme.colorScheme.outlineVariant,
                      height: 1,
                    ),
            ),
            actions: [
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8),
                child: Tooltip(
                  message: _kSwitchModelTooltip,
                  child: OutlinedButton.icon(
                    style: OutlinedButton.styleFrom(
                      foregroundColor: state.hasModel
                          ? theme.colorScheme.onSurface
                          : theme.colorScheme.primary,
                      side: BorderSide(
                        color: state.hasModel
                            ? theme.colorScheme.outline
                            : theme.colorScheme.primary.withValues(alpha: 0.5),
                      ),
                      padding: const EdgeInsets.symmetric(horizontal: 10),
                      visualDensity: VisualDensity.compact,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                    onPressed: () => _pickBundle(state),
                    icon: Icon(
                      state.hasModel
                          ? Icons.memory_rounded
                          : Icons.cloud_download_outlined,
                      size: 14,
                      color: theme.colorScheme.primary,
                    ),
                    label: ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 140),
                      child: Text(
                        loadedModelName ?? _kSelectModelLabel,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          fontSize: 11.5,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 6),
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8),
                child: SegmentedButton<AppUIMode>(
                  style: SegmentedButton.styleFrom(
                    visualDensity: VisualDensity.compact,
                    padding: const EdgeInsets.symmetric(horizontal: 6),
                    side: BorderSide(color: theme.colorScheme.outlineVariant),
                    selectedBackgroundColor: theme.colorScheme.primary
                        .withValues(alpha: 0.16),
                    selectedForegroundColor: theme.colorScheme.primary,
                  ),
                  showSelectedIcon: false,
                  segments: const [
                    ButtonSegment<AppUIMode>(
                      value: AppUIMode.chat,
                      label: Text(
                        _kChatSegmentLabel,
                        style: TextStyle(fontSize: 11.5),
                      ),
                      icon: Icon(Icons.chat_outlined, size: 14),
                    ),
                    ButtonSegment<AppUIMode>(
                      value: AppUIMode.ttsStudio,
                      label: Text(
                        _kTtsStudioSegmentLabel,
                        style: TextStyle(fontSize: 11.5),
                      ),
                      icon: Icon(Icons.record_voice_over_outlined, size: 14),
                    ),
                  ],
                  selected: {state.uiMode},
                  onSelectionChanged: (selection) {
                    _controller.dispatch(SetUIModeIntent(selection.first));
                  },
                ),
              ),
              IconButton(
                icon: const Icon(Icons.delete_sweep_rounded),
                tooltip: _kClearTranscriptTooltip,
                onPressed: state.turns.isEmpty
                    ? null
                    : () => _controller.dispatch(const ClearTranscriptIntent()),
              ),
              IconButton(
                icon: const Icon(Icons.settings_outlined),
                tooltip: _kSettingsTooltip,
                onPressed: () => _showSettingsSheet(state),
              ),
            ],
          ),
          body: state.uiMode == AppUIMode.ttsStudio
              ? TtsStudioView(
                  state: state,
                  controller: _controller,
                  onOpenCatalog: () => _pickBundle(state),
                )
              : Column(
                  children: [
                    Expanded(
                      child: MessageList(
                        turns: state.turns,
                        scrollController: _scrollController,
                        audioPlayer: _controller.audioPlayer,
                        hasModel: state.hasModel,
                        isLoading: state.isLoading,
                        onOpenCatalog: () => _pickBundle(state),
                        onOpenLocalModel: _pickLocalModel,
                        onSelectPrompt: _onSelectStarterPrompt,
                      ),
                    ),
                    if (state.hasModel && (hasAudioIn || hasAudioOut))
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: theme.colorScheme.surface,
                          border: Border(
                            top: BorderSide(
                              color: theme.colorScheme.outlineVariant,
                            ),
                            bottom: BorderSide(
                              color: theme.colorScheme.outlineVariant,
                            ),
                          ),
                        ),
                        child: Center(
                          child: ConstrainedBox(
                            constraints: const BoxConstraints(maxWidth: 840),
                            child: Row(
                              children: [
                                Icon(
                                  Icons.graphic_eq_rounded,
                                  size: 15,
                                  color: theme.colorScheme.primary,
                                ),
                                const SizedBox(width: 6),
                                Text(
                                  _kVoiceModeBarLabel,
                                  style: TextStyle(
                                    fontSize: 11.5,
                                    fontWeight: FontWeight.w600,
                                    color: theme.colorScheme.onSurfaceVariant,
                                  ),
                                ),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: SingleChildScrollView(
                                    scrollDirection: Axis.horizontal,
                                    child: Row(
                                      children: [
                                        if (hasAudioIn && hasAudioOut) ...[
                                          _AudioModeChip(
                                            label: _kVoiceModeChatShortLabel,
                                            icon: Icons
                                                .record_voice_over_outlined,
                                            isSelected:
                                                state.settings.audioChatMode ==
                                                AudioChatMode.interleaved,
                                            onTap: () => _controller.dispatch(
                                              const UpdateSettingsIntent(
                                                audioChatMode:
                                                    AudioChatMode.interleaved,
                                              ),
                                            ),
                                          ),
                                          const SizedBox(width: 6),
                                        ],
                                        if (hasAudioIn) ...[
                                          _AudioModeChip(
                                            label: _kVoiceModeAsrLabel,
                                            icon: Icons.transcribe_rounded,
                                            isSelected:
                                                state.settings.audioChatMode ==
                                                AudioChatMode.speechToText,
                                            onTap: () => _controller.dispatch(
                                              const UpdateSettingsIntent(
                                                audioChatMode:
                                                    AudioChatMode.speechToText,
                                              ),
                                            ),
                                          ),
                                          const SizedBox(width: 6),
                                        ],
                                        if (hasAudioOut) ...[
                                          _AudioModeChip(
                                            label: _kVoiceModeTtsLabel,
                                            icon: Icons.volume_up_outlined,
                                            isSelected:
                                                state.settings.audioChatMode ==
                                                AudioChatMode.textToSpeech,
                                            onTap: () => _controller.dispatch(
                                              const UpdateSettingsIntent(
                                                audioChatMode:
                                                    AudioChatMode.textToSpeech,
                                              ),
                                            ),
                                          ),
                                          const SizedBox(width: 6),
                                        ],
                                        _AudioModeChip(
                                          label: _kVoiceModeTextOnlyChipLabel,
                                          icon:
                                              Icons.chat_bubble_outline_rounded,
                                          isSelected:
                                              state.settings.audioChatMode ==
                                              AudioChatMode.textOnly,
                                          onTap: () => _controller.dispatch(
                                            const UpdateSettingsIntent(
                                              audioChatMode:
                                                  AudioChatMode.textOnly,
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    MessageComposer(
                      controller: _inputController,
                      isBusy: state.isBusy,
                      isGenerating: state.isGenerating,
                      canAttachImage: state.canAttachImage,
                      canAttachAudio: state.canAttachAudio,
                      pendingImageBytes: state.pendingImageBytes,
                      pendingImageName: state.pendingImageName,
                      onSend: _sendMessage,
                      onStop: () =>
                          _controller.dispatch(const StopGenerationIntent()),
                      onPickImage: _pickImage,
                      onClearImage: () => _controller.dispatch(
                        const ClearAttachedImageIntent(),
                      ),
                      onSendAudio: (pcm, sampleRate) {
                        final text = _inputController.text.trim();
                        _inputController.clear();
                        _controller.dispatch(
                          SendAudioPromptIntent(
                            pcmSamples: pcm,
                            sampleRate: sampleRate,
                            prompt: text,
                          ),
                        );
                      },
                    ),
                  ],
                ),
        );
      },
    );
  }
}

class _AudioModeChip extends StatelessWidget {
  const _AudioModeChip({
    required this.label,
    required this.icon,
    required this.isSelected,
    required this.onTap,
  });

  final String label;
  final IconData icon;
  final bool isSelected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(6),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(
          color: isSelected
              ? theme.colorScheme.primary.withValues(alpha: 0.16)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(6),
          border: Border.all(
            color: isSelected
                ? theme.colorScheme.primary.withValues(alpha: 0.4)
                : theme.colorScheme.outlineVariant,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 13,
              color: isSelected
                  ? theme.colorScheme.primary
                  : theme.colorScheme.onSurfaceVariant,
            ),
            const SizedBox(width: 5),
            Text(
              label,
              style: TextStyle(
                fontSize: 11,
                fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
                color: isSelected
                    ? theme.colorScheme.primary
                    : theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
