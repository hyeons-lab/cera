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
import 'package:shared_preferences/shared_preferences.dart';

import 'package:cera_ffi_flutter/cera_ffi_flutter.dart' hide ModelSource;

import 'app_constants.dart';
import 'chat_controller.dart';
import 'chat_intent.dart';
import 'chat_state.dart';
import 'model_source.dart';
import 'widgets/bundle_picker_dialog.dart';
import 'widgets/message_composer.dart';
import 'widgets/message_list.dart';
import 'widgets/voice_persona_picker.dart';

void main() => runApp(const CeraExampleApp());

class CeraExampleApp extends StatefulWidget {
  const CeraExampleApp({super.key});

  @override
  State<CeraExampleApp> createState() => _CeraExampleAppState();
}

class _CeraExampleAppState extends State<CeraExampleApp> {
  // Light palette (Warm Alabaster & Zinc)
  static const _kLightCanvas = Color(0xFFFAFAFA);
  static const _kLightSurface = Color(0xFFFFFFFF);
  static const _kLightContainerLow = Color(0xFFF2F3F5);
  static const _kLightContainerHigh = Color(0xFFE8EAEE);
  static const _kLightTextPrimary = Color(0xFF141417);
  static const _kLightTextSecondary = Color(0xFF646872);
  static const _kLightPrimary = Color(0xFF18181B);
  static const _kLightOutline = Color(0xFFD4D4D8);
  static const _kLightOutlineVariant = Color(0xFFE4E4E7);

  // Dark palette (Clean Charcoal & Zinc)
  static const _kDarkCanvas = Color(0xFF121215);
  static const _kDarkSurface = Color(0xFF18181C);
  static const _kDarkContainerLow = Color(0xFF1D1D22);
  static const _kDarkContainerHigh = Color(0xFF27272E);
  static const _kDarkTextPrimary = Color(0xFFF4F4F5);
  static const _kDarkTextSecondary = Color(0xFF9CA3AF);
  static const _kDarkPrimary = Color(0xFFF4F4F5);
  static const _kDarkOutline = Color(0xFF3F3F46);
  static const _kDarkOutlineVariant = Color(0xFF27272A);

  ThemeMode _themeMode = ThemeMode.dark;

  @override
  void initState() {
    super.initState();
    _loadSavedThemeMode();
  }

  Future<void> _loadSavedThemeMode() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final saved = prefs.getString(PrefKeys.themeMode);
      if (!mounted || saved == null) return;
      setState(() {
        if (saved == PrefValues.themeLight) {
          _themeMode = ThemeMode.light;
        } else if (saved == PrefValues.themeDark) {
          _themeMode = ThemeMode.dark;
        } else if (saved == PrefValues.themeSystem) {
          _themeMode = ThemeMode.system;
        }
      });
    } catch (_) {}
  }

  Future<void> _toggleThemeMode(Brightness currentBrightness) async {
    final nextMode = currentBrightness == Brightness.dark
        ? ThemeMode.light
        : ThemeMode.dark;
    setState(() => _themeMode = nextMode);
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(
        PrefKeys.themeMode,
        nextMode == ThemeMode.light
            ? PrefValues.themeLight
            : PrefValues.themeDark,
      );
    } catch (_) {}
  }

  static ThemeData _buildLightTheme() {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.light,
      scaffoldBackgroundColor: _kLightCanvas,
      canvasColor: _kLightCanvas,
      cardColor: _kLightSurface,
      dividerColor: _kLightOutlineVariant,
      colorScheme: const ColorScheme.light(
        primary: _kLightPrimary,
        onPrimary: _kLightSurface,
        primaryContainer: _kLightContainerHigh,
        onPrimaryContainer: _kLightTextPrimary,
        surface: _kLightSurface,
        surfaceContainerLow: _kLightContainerLow,
        surfaceContainerHigh: _kLightContainerHigh,
        onSurface: _kLightTextPrimary,
        onSurfaceVariant: _kLightTextSecondary,
        outline: _kLightOutline,
        outlineVariant: _kLightOutlineVariant,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: _kLightCanvas,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        scrolledUnderElevation: 0,
      ),
    );
  }

  static ThemeData _buildDarkTheme() {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: _kDarkCanvas,
      canvasColor: _kDarkCanvas,
      cardColor: _kDarkSurface,
      dividerColor: _kDarkOutlineVariant,
      colorScheme: const ColorScheme.dark(
        primary: _kDarkPrimary,
        onPrimary: _kDarkCanvas,
        primaryContainer: _kDarkContainerHigh,
        onPrimaryContainer: _kDarkTextPrimary,
        surface: _kDarkSurface,
        surfaceContainerLow: _kDarkContainerLow,
        surfaceContainerHigh: _kDarkContainerHigh,
        onSurface: _kDarkTextPrimary,
        onSurfaceVariant: _kDarkTextSecondary,
        outline: _kDarkOutline,
        outlineVariant: _kDarkOutlineVariant,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: _kDarkCanvas,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        scrolledUnderElevation: 0,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: AppStrings.appTitle,
      debugShowCheckedModeBanner: false,
      themeMode: _themeMode,
      theme: _buildLightTheme(),
      darkTheme: _buildDarkTheme(),
      home: ChatPage(onToggleTheme: _toggleThemeMode),
    );
  }
}

/// Main chat interface built with MVI architecture.
class ChatPage extends StatefulWidget {
  const ChatPage({super.key, this.onToggleTheme});

  final ValueChanged<Brightness>? onToggleTheme;

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  static const _kPickModelDialogTitle = 'Choose a .gguf model';
  static const _kPickImageDialogTitle = 'Select an image to attach';
  static const _kUnreadableImageBytesError = 'Could not read image file bytes';

  // Top bar labels & tooltips
  static const _kSwitchModelTooltip = 'Select or download model';
  static const _kSelectModelLabel = 'Select Model';
  static const _kClearTranscriptTooltip = 'New chat';
  static const _kToggleThemeTooltip = 'Toggle light or dark theme';
  static const _kSettingsTooltip = 'Tune & settings';
  static const _kStatusEmerald = Color(0xFF10B981);
  static const _kStatusAmber = Color(0xFFF59E0B);

  // Settings sheet section headers & copy
  static const _kSettingsTitle = 'Tune & Settings';
  static const _kSectionModels = 'MODEL';
  static const _kCatalogModelsTitle = 'Browse Model Catalog';
  static const _kCatalogModelsSubtitle =
      'Switch between cached models or download new ones';
  static const _kOpenLocalTitle = 'Open Local .gguf File';
  static const _kOpenLocalSubtitle = 'Pick a model file from disk storage';
  static const _kUnloadModelTitle = 'Unload Active Model';
  static const _kUnloadModelSubtitle = 'Release memory and close model engine';

  static const _kSectionSampling = 'SAMPLING & GENERATION';
  static const _kSamplingTemperatureTitle = 'Temperature';
  static const _kSamplingAutoHint =
      'Auto uses model defaults (0.0 greedy argmax on WebGPU text, 0.7 on audio).';
  static const _kSamplingGreedyHint =
      'Greedy (0.0): deterministic argmax decoding with GPU fast path.';
  static const _kSamplingStochasticHint =
      'Stochastic (> 0.0): nucleus / top-k sampling for varied responses.';
  static const _kPresetGreedyLabel = 'Greedy (0.0)';
  static const _kMaxOutputTokensTitle = 'Max Output Tokens';
  static const _kMaxOutputTokensSubtitle =
      'Maximum tokens generated per assistant response';

  static const _kSectionInference = 'EXECUTION & MULTIMODAL';
  static const _kComputeBackendTitle = 'Compute Backend';
  static const _kComputeBackendSubtitle =
      'Auto, GPU (WebGPU / Metal), NPU (Hexagon), or CPU';
  static const _kBackendAutoLabel = 'Auto';
  static const _kBackendGpuLabel = 'GPU (WebGPU / Metal)';
  static const _kBackendHexagonLabel = 'NPU (Hexagon)';
  static const _kBackendCpuLabel = 'CPU';
  static const _kTurboQuantTitle = 'TurboQuant KV Compression';
  static const _kTurboQuantSubtitle =
      '3-bit keys / 2-bit values for lower memory footprint';
  static const _kVisionMaxDimTitle = 'Vision Image Resolution';
  static const _kVisionMaxDimSubtitle =
      'Max long-side dimension before ViT patch encoding';
  static const _kImageDimFastLabel = '256 px (Fast)';
  static const _kImageDimHdLabel = '512 px (HD)';
  static const _kImageDimNativeLabel = 'Native';

  static const _kSectionVoice = 'VOICE & SPEECH';
  static const _kVoiceModeTitle = 'Audio Mode';
  static const _kVoiceModeSubtitle =
      'How audio models handle speech input and output';
  static const _kVoiceModeAsrLabel = 'Speech to Text (ASR)';
  static const _kVoiceModeInterleavedLabel = 'Voice Chat';
  static const _kVoiceModeTtsLabel = 'Text to Speech (TTS)';
  static const _kVoiceModeTextOnlyMenuLabel = 'Text Only';
  static const _kVoicePersonaTitle = 'Voice Persona';
  static const _kVoicePersonaUnavailableSubtitle =
      'Load an audio generation model (such as LFM2.5-Audio-1.5B) to select a speaker voice';
  static const _kVoicePersonaUnavailableBadge = 'No Audio Model';
  static const _kActiveRuntimePrefix = 'Active backend: ';

  static const _kModelPlaceholderToken = '{model}';

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
    _controller.dispatch(const RestoreLastModelIntent());
  }

  void _onStateChange() {
    if (!mounted) return;
    if (_controller.value.isGenerating) {
      if (!_scrollController.hasClients) return;
      final pos = _scrollController.position;
      final distanceToBottom = pos.maxScrollExtent - pos.pixels;
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

  Future<String?> _storeDir() async {
    if (kIsWeb) return null;
    final mobile =
        defaultTargetPlatform == TargetPlatform.android ||
        defaultTargetPlatform == TargetPlatform.iOS;
    if (!mobile) return null;
    return (await getApplicationSupportDirectory()).path;
  }

  Future<void> _pickBundle(ChatState state) async {
    final loaded = state.loadedModel;
    final bundleModel = loaded is BundleModelSource ? loaded : null;

    final choice = await showDialog<BundleChoice>(
      context: context,
      builder: (_) => BundlePickerDialog(
        currentBundleName: bundleModel?.bundleName,
        currentQuant: bundleModel?.quant,
        onOpenLocalModel: _pickLocalModel,
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

  List<String> _resolvedTtsSamples(ChatState state) {
    final rawModelName = state.loadedModel?.name ?? AppStrings.appTitle;
    final cleanModelName = rawModelName
        .split(AppStrings.metaSeparator)
        .first
        .trim();
    final profileSamples = state.audioProfile?.sampleTexts ?? const <String>[];
    if (profileSamples.isEmpty) return const <String>[];
    return [
      for (final s in profileSamples)
        s.replaceAll(_kModelPlaceholderToken, cleanModelName),
    ];
  }

  void _showSettingsSheet(ChatState state) {
    final theme = Theme.of(context);
    showModalBottomSheet(
      context: context,
      backgroundColor: theme.colorScheme.surface,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
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
                padding: const EdgeInsets.fromLTRB(20, 12, 20, 20),
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
                            Icons.tune_rounded,
                            size: 20,
                            color: theme.colorScheme.onSurface,
                          ),
                          const SizedBox(width: 8),
                          Text(
                            _kSettingsTitle,
                            style: TextStyle(
                              fontSize: 17,
                              fontWeight: FontWeight.w600,
                              color: theme.colorScheme.onSurface,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 18),
                      _SectionLabel(label: _kSectionModels, theme: theme),
                      const SizedBox(height: 8),
                      _TonalCard(
                        theme: theme,
                        child: Column(
                          children: [
                            if (currentState.hasModel)
                              ListTile(
                                leading: const Icon(
                                  Icons.memory_rounded,
                                  color: _kStatusEmerald,
                                ),
                                title: Text(
                                  currentState.loadedModel?.name ??
                                      AppStrings.unknownModel,
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                  style: const TextStyle(
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                                subtitle: Text(
                                  '$_kActiveRuntimePrefix${currentState.backend ?? AppStrings.unknownBackend}',
                                ),
                                trailing: Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 9,
                                    vertical: 4,
                                  ),
                                  decoration: BoxDecoration(
                                    color:
                                        theme.colorScheme.surfaceContainerHigh,
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                  child: Text(
                                    currentState.backend ??
                                        AppStrings.unknownBackend,
                                    style: TextStyle(
                                      fontSize: 11.5,
                                      fontWeight: FontWeight.w600,
                                      color: theme.colorScheme.onSurface,
                                    ),
                                  ),
                                ),
                              ),
                            ListTile(
                              leading: Icon(
                                Icons.cloud_download_outlined,
                                color: theme.colorScheme.onSurface,
                              ),
                              title: const Text(_kCatalogModelsTitle),
                              subtitle: const Text(_kCatalogModelsSubtitle),
                              onTap: () {
                                Navigator.of(context).pop();
                                _pickBundle(_controller.value);
                              },
                            ),
                            ListTile(
                              leading: Icon(
                                Icons.folder_open_outlined,
                                color: theme.colorScheme.onSurface,
                              ),
                              title: const Text(_kOpenLocalTitle),
                              subtitle: const Text(_kOpenLocalSubtitle),
                              onTap: () {
                                Navigator.of(context).pop();
                                _pickLocalModel();
                              },
                            ),
                            if (currentState.hasModel)
                              ListTile(
                                leading: Icon(
                                  Icons.eject_outlined,
                                  color: theme.colorScheme.error,
                                ),
                                title: Text(
                                  _kUnloadModelTitle,
                                  style: TextStyle(
                                    color: theme.colorScheme.error,
                                  ),
                                ),
                                subtitle: const Text(_kUnloadModelSubtitle),
                                onTap: () {
                                  Navigator.of(context).pop();
                                  _controller.dispatch(
                                    const UnloadModelIntent(),
                                  );
                                },
                              ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 18),
                      _SectionLabel(label: _kSectionSampling, theme: theme),
                      const SizedBox(height: 8),
                      _TonalCard(
                        theme: theme,
                        padding: const EdgeInsets.all(16),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Flexible(
                                  child: Text(
                                    _kSamplingTemperatureTitle,
                                    overflow: TextOverflow.ellipsis,
                                    style: TextStyle(
                                      fontSize: 14.5,
                                      fontWeight: FontWeight.w500,
                                      color: theme.colorScheme.onSurface,
                                    ),
                                  ),
                                ),
                                const SizedBox(width: 8),
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 9,
                                    vertical: 3,
                                  ),
                                  decoration: BoxDecoration(
                                    color:
                                        theme.colorScheme.surfaceContainerHigh,
                                    borderRadius: BorderRadius.circular(10),
                                  ),
                                  child: Text(
                                    currentSettings.samplingLabel,
                                    style: TextStyle(
                                      fontSize: 12,
                                      fontWeight: FontWeight.w600,
                                      color: theme.colorScheme.onSurface,
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
                                    style: TextStyle(fontSize: 12),
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
                                    style: TextStyle(fontSize: 12),
                                  ),
                                  selected:
                                      activeTemp != null &&
                                      (activeTemp -
                                                  ChatDefaults
                                                      .greedyTemperature)
                                              .abs() <
                                          0.01,
                                  onSelected: (_) => _controller.dispatch(
                                    const UpdateSettingsIntent(
                                      temperature:
                                          ChatDefaults.greedyTemperature,
                                    ),
                                  ),
                                ),
                                ChoiceChip(
                                  label: const Text(
                                    AppStrings.samplingBalanced,
                                    style: TextStyle(fontSize: 12),
                                  ),
                                  selected:
                                      activeTemp != null &&
                                      (activeTemp -
                                                  ChatDefaults
                                                      .balancedTemperature)
                                              .abs() <
                                          0.01,
                                  onSelected: (_) => _controller.dispatch(
                                    const UpdateSettingsIntent(
                                      temperature:
                                          ChatDefaults.balancedTemperature,
                                    ),
                                  ),
                                ),
                                ChoiceChip(
                                  label: const Text(
                                    AppStrings.samplingCreative,
                                    style: TextStyle(fontSize: 12),
                                  ),
                                  selected:
                                      activeTemp != null &&
                                      (activeTemp -
                                                  ChatDefaults
                                                      .creativeTemperature)
                                              .abs() <
                                          0.01,
                                  onSelected: (_) => _controller.dispatch(
                                    const UpdateSettingsIntent(
                                      temperature:
                                          ChatDefaults.creativeTemperature,
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
                                  for (final count
                                      in ChatDefaults.maxTokenChoices)
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
                          ],
                        ),
                      ),
                      const SizedBox(height: 18),
                      _SectionLabel(label: _kSectionVoice, theme: theme),
                      const SizedBox(height: 8),
                      _TonalCard(
                        theme: theme,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 8,
                        ),
                        child: Column(
                          children: [
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
                              trailing: ConstrainedBox(
                                constraints: const BoxConstraints(
                                  maxWidth: 150,
                                ),
                                child: DropdownButton<AudioChatMode>(
                                  value: currentSettings.audioChatMode,
                                  isExpanded: true,
                                  dropdownColor: theme.colorScheme.surface,
                                  underline: const SizedBox.shrink(),
                                  items: const [
                                    DropdownMenuItem(
                                      value: AudioChatMode.speechToText,
                                      child: Text(
                                        _kVoiceModeAsrLabel,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                    DropdownMenuItem(
                                      value: AudioChatMode.interleaved,
                                      child: Text(
                                        _kVoiceModeInterleavedLabel,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                    DropdownMenuItem(
                                      value: AudioChatMode.textToSpeech,
                                      child: Text(
                                        _kVoiceModeTtsLabel,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                    DropdownMenuItem(
                                      value: AudioChatMode.textOnly,
                                      child: Text(
                                        _kVoiceModeTextOnlyMenuLabel,
                                        overflow: TextOverflow.ellipsis,
                                      ),
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
                            ),
                            if (currentState.audioProfile?.hasVoices ?? false)
                              VoicePersonaPicker(
                                profile: currentState.audioProfile,
                                saved: currentSettings.chatVoice,
                                onChanged: (val) {
                                  final hasAudioOut =
                                      currentState.capabilities?.audioOut ??
                                      false;
                                  final hasAudioIn =
                                      currentState.capabilities?.audioIn ??
                                      false;
                                  final shouldEnableSpeech =
                                      hasAudioOut &&
                                      currentSettings.audioChatMode !=
                                          AudioChatMode.interleaved &&
                                      currentSettings.audioChatMode !=
                                          AudioChatMode.textToSpeech;
                                  _controller.dispatch(
                                    UpdateSettingsIntent(
                                      chatVoice: val,
                                      clearChatVoice: val == null,
                                      audioChatMode: shouldEnableSpeech
                                          ? (hasAudioIn
                                                ? AudioChatMode.interleaved
                                                : AudioChatMode.textToSpeech)
                                          : null,
                                    ),
                                  );
                                },
                              )
                            else
                              ListTile(
                                contentPadding: EdgeInsets.zero,
                                title: const Text(_kVoicePersonaTitle),
                                subtitle: Text(
                                  _kVoicePersonaUnavailableSubtitle,
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: theme.colorScheme.onSurfaceVariant,
                                  ),
                                ),
                                trailing: Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 9,
                                    vertical: 4,
                                  ),
                                  decoration: BoxDecoration(
                                    color:
                                        theme.colorScheme.surfaceContainerHigh,
                                    borderRadius: BorderRadius.circular(10),
                                  ),
                                  child: Text(
                                    _kVoicePersonaUnavailableBadge,
                                    style: TextStyle(
                                      fontSize: 11,
                                      fontWeight: FontWeight.w500,
                                      color: theme.colorScheme.onSurfaceVariant,
                                    ),
                                  ),
                                ),
                              ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 18),
                      _SectionLabel(label: _kSectionInference, theme: theme),
                      const SizedBox(height: 8),
                      _TonalCard(
                        theme: theme,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 8,
                        ),
                        child: Column(
                          children: [
                            ListTile(
                              contentPadding: EdgeInsets.zero,
                              title: const Text(_kComputeBackendTitle),
                              subtitle: Text(
                                currentState.backend != null &&
                                        currentState.backend!.isNotEmpty
                                    ? '$_kActiveRuntimePrefix${currentState.backend}${AppStrings.metaSeparator}$_kComputeBackendSubtitle'
                                    : _kComputeBackendSubtitle,
                                style: TextStyle(
                                  fontSize: 12,
                                  color: theme.colorScheme.onSurfaceVariant,
                                ),
                              ),
                              trailing: ConstrainedBox(
                                constraints: const BoxConstraints(
                                  maxWidth: 150,
                                ),
                                child: DropdownButton<CeraBackend>(
                                  value: currentSettings.backend,
                                  isExpanded: true,
                                  dropdownColor: theme.colorScheme.surface,
                                  underline: const SizedBox.shrink(),
                                  items: const [
                                    DropdownMenuItem(
                                      value: CeraBackend.auto,
                                      child: Text(
                                        _kBackendAutoLabel,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                    DropdownMenuItem(
                                      value: CeraBackend.gpu,
                                      child: Text(
                                        _kBackendGpuLabel,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                    DropdownMenuItem(
                                      value: CeraBackend.hexagon,
                                      child: Text(
                                        _kBackendHexagonLabel,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                    DropdownMenuItem(
                                      value: CeraBackend.cpu,
                                      child: Text(
                                        _kBackendCpuLabel,
                                        overflow: TextOverflow.ellipsis,
                                      ),
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
                            ),
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
                              trailing: ConstrainedBox(
                                constraints: const BoxConstraints(
                                  maxWidth: 135,
                                ),
                                child: DropdownButton<int?>(
                                  value: currentSettings.maxImageLongSize,
                                  isExpanded: true,
                                  dropdownColor: theme.colorScheme.surface,
                                  underline: const SizedBox.shrink(),
                                  items: const [
                                    DropdownMenuItem(
                                      value: ChatDefaults.maxImageLongSize,
                                      child: Text(
                                        _kImageDimFastLabel,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                    DropdownMenuItem(
                                      value: ChatDefaults.hdImageLongSize,
                                      child: Text(
                                        _kImageDimHdLabel,
                                        overflow: TextOverflow.ellipsis,
                                      ),
                                    ),
                                    DropdownMenuItem(
                                      value: null,
                                      child: Text(
                                        _kImageDimNativeLabel,
                                        overflow: TextOverflow.ellipsis,
                                      ),
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
                            ),
                          ],
                        ),
                      ),
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

  PreferredSizeWidget _buildMinimalistAppBar(
    BuildContext context,
    ThemeData theme,
    ChatState state,
    String? loadedModelName,
  ) {
    final isDark = theme.brightness == Brightness.dark;
    final activeBackend = state.backend;
    final statusColor = state.isLoading
        ? _kStatusAmber
        : (state.hasModel
              ? _kStatusEmerald
              : theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.45));
    final tooltipText = activeBackend != null && activeBackend.isNotEmpty
        ? '$_kActiveRuntimePrefix$activeBackend${AppStrings.metaSeparator}$_kSwitchModelTooltip'
        : _kSwitchModelTooltip;

    return AppBar(
      automaticallyImplyLeading: false,
      toolbarHeight: 56,
      titleSpacing: 8,
      centerTitle: true,
      leadingWidth: 52,
      leading: Padding(
        padding: const EdgeInsets.only(left: 8),
        child: IconButton(
          icon: Icon(
            Icons.delete_sweep_rounded,
            size: 20,
            color: state.turns.isEmpty
                ? theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.35)
                : theme.colorScheme.onSurfaceVariant,
          ),
          tooltip: _kClearTranscriptTooltip,
          onPressed: state.turns.isEmpty
              ? null
              : () => _controller.dispatch(const ClearTranscriptIntent()),
        ),
      ),
      title: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 340),
        child: Tooltip(
          message: tooltipText,
          child: Material(
            color: theme.colorScheme.surfaceContainerLow,
            borderRadius: BorderRadius.circular(20),
            child: InkWell(
              onTap: () => _pickBundle(state),
              borderRadius: BorderRadius.circular(20),
              child: Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 7,
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 7,
                      height: 7,
                      decoration: BoxDecoration(
                        color: statusColor,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Flexible(
                      child: Text(
                        loadedModelName ?? _kSelectModelLabel,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: theme.colorScheme.onSurface,
                        ),
                      ),
                    ),
                    if (activeBackend != null && activeBackend.isNotEmpty) ...[
                      const SizedBox(width: 6),
                      ConstrainedBox(
                        constraints: const BoxConstraints(maxWidth: 88),
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 6,
                            vertical: 2,
                          ),
                          decoration: BoxDecoration(
                            color: theme.colorScheme.surfaceContainerHigh,
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: Text(
                            activeBackend,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: TextStyle(
                              fontSize: 10.5,
                              fontWeight: FontWeight.w600,
                              color: theme.colorScheme.onSurfaceVariant,
                            ),
                          ),
                        ),
                      ),
                    ],
                    const SizedBox(width: 4),
                    Icon(
                      Icons.keyboard_arrow_down_rounded,
                      size: 16,
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
      actions: [
        if (widget.onToggleTheme != null)
          IconButton(
            icon: Icon(
              isDark ? Icons.light_mode_outlined : Icons.dark_mode_outlined,
              size: 19,
              color: theme.colorScheme.onSurfaceVariant,
            ),
            tooltip: _kToggleThemeTooltip,
            visualDensity: VisualDensity.compact,
            onPressed: () => widget.onToggleTheme?.call(theme.brightness),
          ),
        Padding(
          padding: const EdgeInsets.only(right: 8),
          child: IconButton(
            icon: Icon(
              Icons.settings_outlined,
              size: 19,
              color: theme.colorScheme.onSurfaceVariant,
            ),
            tooltip: _kSettingsTooltip,
            visualDensity: VisualDensity.compact,
            onPressed: () => _showSettingsSheet(state),
          ),
        ),
      ],
      bottom: state.isLoading
          ? PreferredSize(
              preferredSize: const Size.fromHeight(2),
              child: LinearProgressIndicator(
                value: state.downloadFraction,
                minHeight: 2,
                backgroundColor: theme.colorScheme.surfaceContainerLow,
                valueColor: AlwaysStoppedAnimation<Color>(
                  theme.colorScheme.primary,
                ),
              ),
            )
          : null,
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
        final isTtsMode =
            hasAudioOut &&
            state.settings.audioChatMode == AudioChatMode.textToSpeech;
        final ttsSamples = _resolvedTtsSamples(state);

        void onSelectVoice(String? voice) {
          final shouldEnableSpeech =
              hasAudioOut &&
              state.settings.audioChatMode != AudioChatMode.interleaved &&
              state.settings.audioChatMode != AudioChatMode.textToSpeech;
          _controller.dispatch(
            UpdateSettingsIntent(
              chatVoice: voice,
              clearChatVoice: voice == null,
              audioChatMode: shouldEnableSpeech
                  ? (hasAudioIn
                        ? AudioChatMode.interleaved
                        : AudioChatMode.textToSpeech)
                  : null,
            ),
          );
        }

        return Scaffold(
          appBar: _buildMinimalistAppBar(
            context,
            theme,
            state,
            loadedModelName,
          ),
          body: Column(
            children: [
              Expanded(
                child: MessageList(
                  turns: state.turns,
                  scrollController: _scrollController,
                  audioPlayer: _controller.audioPlayer,
                  hasModel: state.hasModel,
                  isLoading: state.isLoading,
                  statusText: state.status,
                  activeBackend: state.backend,
                  isTtsMode: isTtsMode,
                  ttsSampleTexts: ttsSamples,
                  audioProfile: state.audioProfile,
                  selectedVoicePrompt: state.settings.chatVoice,
                  onVoiceChanged: onSelectVoice,
                  onOpenCatalog: () => _pickBundle(state),
                  onOpenLocalModel: _pickLocalModel,
                  onSelectPrompt: _onSelectStarterPrompt,
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
                hasAudioIn: hasAudioIn,
                hasAudioOut: hasAudioOut,
                audioChatMode: state.settings.audioChatMode,
                onAudioChatModeChanged: (mode) => _controller.dispatch(
                  UpdateSettingsIntent(audioChatMode: mode),
                ),
                audioProfile: state.audioProfile,
                selectedVoicePrompt: state.settings.chatVoice,
                onVoiceChanged: onSelectVoice,
                ttsSampleTexts: ttsSamples,
                onSend: _sendMessage,
                onStop: () =>
                    _controller.dispatch(const StopGenerationIntent()),
                onPickImage: _pickImage,
                onClearImage: () =>
                    _controller.dispatch(const ClearAttachedImageIntent()),
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

class _SectionLabel extends StatelessWidget {
  const _SectionLabel({required this.label, required this.theme});

  final String label;
  final ThemeData theme;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 4),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.8,
          color: theme.colorScheme.onSurfaceVariant,
        ),
      ),
    );
  }
}

class _TonalCard extends StatelessWidget {
  const _TonalCard({required this.theme, required this.child, this.padding});

  final ThemeData theme;
  final Widget child;
  final EdgeInsetsGeometry? padding;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerLow,
        borderRadius: BorderRadius.circular(18),
      ),
      child: child,
    );
  }
}
