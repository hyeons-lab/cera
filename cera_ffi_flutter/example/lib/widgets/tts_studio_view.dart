import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../app_constants.dart';
import '../chat_controller.dart';
import '../chat_intent.dart';
import '../chat_state.dart';
import '../model_source.dart';
import 'audio_waveform.dart';

/// Dedicated UI for on-device Neural Text-to-Speech synthesis using Cera's vocoder.
class TtsStudioView extends StatefulWidget {
  const TtsStudioView({
    super.key,
    required this.state,
    required this.controller,
    required this.onOpenCatalog,
  });

  final ChatState state;
  final ChatController controller;
  final VoidCallback onOpenCatalog;

  @override
  State<TtsStudioView> createState() => _TtsStudioViewState();
}

class _TtsStudioViewState extends State<TtsStudioView> {
  static const _kDefaultModelFallback = 'LFM2';
  static const _kModelPlaceholder = '{model}';
  static const _kStudioTitle = 'Neural Text-to-Speech Studio';
  static const _kStudioSubtitle =
      'Direct on-device speech synthesis via neural vocoder (no external TTS library).';
  static const _kDefaultBackendFallback = 'GPU';
  static const _kAudioModelRequiredHint =
      'An audio model (like LFM2.5-Audio) is required for neural TTS.';
  static const _kLoadAudioModelLabel = 'Load Audio Model';
  static const _kSampleScriptsHeader = 'SAMPLE SCRIPTS';
  static const _kInputTextHeader = 'INPUT TEXT TO SPEAK';
  static const _kInputTextHint = 'Enter text to synthesize into speech...';
  static const _kVoicePersonaHeader = 'VOICE PERSONA';
  static const _kSynthesizingLabel = 'Synthesizing...';
  static const _kSynthesizeLabel = 'Synthesize Speech';
  static const _kStopLabel = 'Stop';
  static const _kClearLabel = 'Clear';
  static const _kOutputsHeader = 'SYNTHESIZED SPEECH OUTPUTS';
  static const _kClearOutputsLabel = 'Clear Outputs';
  static const _kDefaultSynthesizingStatus = 'Synthesizing audio...';

  static const _genericSamples = [
    'Hello, this voice was synthesized entirely on-device with the $_kModelPlaceholder model powered by Cera.',
    'Cera runs high-performance multimodal AI on Apple Silicon, WebGPU, and CPU.',
    'On-device intelligence guarantees complete privacy with zero cloud latency.',
    'Streaming ISTFT produces 24 kHz neural audio in real-time without external TTS libraries.',
  ];

  late final TextEditingController _textController;

  static String _getModelDisplayName(ChatState state) {
    final model = state.loadedModel;
    final bundleDisplay = model is BundleModelSource ? model.displayName : null;
    if (bundleDisplay != null && bundleDisplay.isNotEmpty) {
      return bundleDisplay;
    }
    final rawName = model?.name ?? '';
    if (rawName.isEmpty) {
      return _kDefaultModelFallback;
    }
    var name = rawName;
    if (name.toLowerCase().endsWith(ModelFormat.ggufExtension)) {
      name = name.substring(0, name.length - ModelFormat.ggufExtension.length);
    }
    if (name.contains(AppStrings.metaSeparator)) {
      name = name.split(AppStrings.metaSeparator).first;
    }
    return name;
  }

  /// Sample sentences for the loaded model: its profile's own, or the generic
  /// English ones.
  static List<String> _getSamplePrompts(ChatState state) {
    final modelName = _getModelDisplayName(state);
    final custom = state.audioProfile?.sampleTexts ?? const <String>[];
    final templates = custom.isNotEmpty ? custom : _genericSamples;
    return [
      for (final t in templates) t.replaceAll(_kModelPlaceholder, modelName),
    ];
  }

  static String _defaultSampleText(ChatState state) =>
      _getSamplePrompts(state).first;

  @override
  void initState() {
    super.initState();
    _textController = TextEditingController(
      text: _defaultSampleText(widget.state),
    );
  }

  @override
  void didUpdateWidget(TtsStudioView oldWidget) {
    super.didUpdateWidget(oldWidget);
    final oldDefault = _defaultSampleText(oldWidget.state);
    final newDefault = _defaultSampleText(widget.state);
    if (oldDefault != newDefault) {
      if (_textController.text.isEmpty || _textController.text == oldDefault) {
        _textController.text = newDefault;
      }
    }
  }

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

  /// The voice that will be sent: the saved one if this model lists it, else
  /// the model's default (a voice saved under another model is not offered).
  static String? _selectedVoice(ChatState state) =>
      state.audioProfile?.voiceFor(state.settings.ttsStudioVoice)?.prompt;

  Future<void> _synthesize() async {
    final text = _textController.text.trim();
    if (text.isEmpty || widget.state.isBusy) return;

    // Ensure audio mode is set to textToSpeech
    if (widget.state.settings.audioChatMode != AudioChatMode.textToSpeech) {
      await widget.controller.dispatch(
        const UpdateSettingsIntent(audioChatMode: AudioChatMode.textToSpeech),
      );
      if (!mounted) return;
    }

    widget.controller.dispatch(SendMessageIntent(text));
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final state = widget.state;
    final hasAudioOut = state.capabilities?.audioOut ?? false;
    final voices = state.audioProfile?.voices ?? const [];

    // Find turns that contain assistant audio outputs
    final audioTurns = state.turns
        .where(
          (t) =>
              t.role == TurnRole.assistant &&
              (t.audioSamples != null ||
                  t.audioDurationSeconds != null ||
                  t.isGenerating),
        )
        .toList();

    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 840),
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header Card
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: theme.colorScheme.surface,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: theme.colorScheme.outlineVariant),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: theme.colorScheme.primary.withValues(
                              alpha: 0.16,
                            ),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Icon(
                            Icons.record_voice_over_rounded,
                            color: theme.colorScheme.primary,
                            size: 20,
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                _kStudioTitle,
                                style: TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.w700,
                                  color: theme.colorScheme.onSurface,
                                ),
                              ),
                              const SizedBox(height: 2),
                              Text(
                                _kStudioSubtitle,
                                style: TextStyle(
                                  fontSize: 12,
                                  color: theme.colorScheme.onSurfaceVariant,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    if (hasAudioOut)
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: theme.colorScheme.primary.withValues(
                            alpha: 0.08,
                          ),
                          borderRadius: BorderRadius.circular(6),
                          border: Border.all(
                            color: theme.colorScheme.primary.withValues(
                              alpha: 0.2,
                            ),
                          ),
                        ),
                        child: Row(
                          children: [
                            Icon(
                              Icons.check_circle_outline_rounded,
                              size: 14,
                              color: theme.colorScheme.primary,
                            ),
                            const SizedBox(width: 6),
                            Expanded(
                              child: Text(
                                'Vocoder Active: 24,000 Hz Float32${AppStrings.metaSeparator}${state.backend ?? _kDefaultBackendFallback}',
                                style: TextStyle(
                                  fontSize: 11.5,
                                  fontWeight: FontWeight.w600,
                                  color: theme.colorScheme.primary,
                                ),
                              ),
                            ),
                          ],
                        ),
                      )
                    else
                      Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: theme.colorScheme.error.withValues(
                            alpha: 0.08,
                          ),
                          borderRadius: BorderRadius.circular(6),
                          border: Border.all(
                            color: theme.colorScheme.error.withValues(
                              alpha: 0.25,
                            ),
                          ),
                        ),
                        child: Row(
                          children: [
                            Icon(
                              Icons.info_outline_rounded,
                              size: 16,
                              color: theme.colorScheme.error,
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                _kAudioModelRequiredHint,
                                style: TextStyle(
                                  fontSize: 12,
                                  color: theme.colorScheme.onSurface,
                                ),
                              ),
                            ),
                            const SizedBox(width: 8),
                            FilledButton.tonal(
                              style: FilledButton.styleFrom(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 10,
                                  vertical: 6,
                                ),
                                visualDensity: VisualDensity.compact,
                              ),
                              onPressed: widget.onOpenCatalog,
                              child: const Text(
                                _kLoadAudioModelLabel,
                                style: TextStyle(fontSize: 11),
                              ),
                            ),
                          ],
                        ),
                      ),
                  ],
                ),
              ),
              const SizedBox(height: 18),

              // Sample Prompts
              Text(
                _kSampleScriptsHeader,
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.8,
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 8),
              SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(
                  children: _getSamplePrompts(state).map((prompt) {
                    return Padding(
                      padding: const EdgeInsets.only(right: 8),
                      child: ActionChip(
                        label: Text(
                          prompt.length > 36
                              ? '${prompt.substring(0, 36)}...'
                              : prompt,
                          style: const TextStyle(fontSize: 11),
                        ),
                        backgroundColor: theme.colorScheme.surface,
                        side: BorderSide(
                          color: theme.colorScheme.outlineVariant,
                        ),
                        onPressed: () {
                          setState(() {
                            _textController.text = prompt;
                          });
                        },
                      ),
                    );
                  }).toList(),
                ),
              ),
              const SizedBox(height: 16),

              // Input Text Box
              Text(
                _kInputTextHeader,
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.8,
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 8),
              TextField(
                controller: _textController,
                maxLines: 4,
                style: TextStyle(
                  fontSize: 14,
                  color: theme.colorScheme.onSurface,
                ),
                decoration: InputDecoration(
                  hintText: _kInputTextHint,
                  hintStyle: TextStyle(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                  filled: true,
                  fillColor: theme.colorScheme.surface,
                  contentPadding: const EdgeInsets.all(14),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10),
                    borderSide: BorderSide(
                      color: theme.colorScheme.outlineVariant,
                    ),
                  ),
                  enabledBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10),
                    borderSide: BorderSide(
                      color: theme.colorScheme.outlineVariant,
                    ),
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10),
                    borderSide: BorderSide(
                      color: theme.colorScheme.primary,
                      width: 1.5,
                    ),
                  ),
                ),
              ),
              // Voice Persona Selector: only for a model whose profile lists voices.
              if (voices.isNotEmpty) ...[
                Text(
                  _kVoicePersonaHeader,
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.8,
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    for (final voice in voices)
                      _VoiceChip(
                        label: voice.label,
                        selected: _selectedVoice(state) == voice.prompt,
                        onSelected: () => widget.controller.dispatch(
                          UpdateSettingsIntent(ttsStudioVoice: voice.prompt),
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 16),
              ],

              // Action Buttons
              Wrap(
                spacing: 10,
                runSpacing: 8,
                alignment: WrapAlignment.spaceBetween,
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Flexible(
                        child: FilledButton.icon(
                          style: FilledButton.styleFrom(
                            backgroundColor: theme.colorScheme.primary,
                            foregroundColor: theme.colorScheme.onPrimary,
                            padding: const EdgeInsets.symmetric(
                              horizontal: 18,
                              vertical: 12,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(8),
                            ),
                          ),
                          icon: state.isGenerating
                              ? SizedBox(
                                  width: 16,
                                  height: 16,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: theme.colorScheme.onPrimary,
                                  ),
                                )
                              : const Icon(Icons.volume_up_rounded, size: 18),
                          label: Text(
                            state.isGenerating
                                ? _kSynthesizingLabel
                                : _kSynthesizeLabel,
                            overflow: TextOverflow.ellipsis,
                            style: const TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          onPressed: (state.isBusy || !hasAudioOut)
                              ? null
                              : _synthesize,
                        ),
                      ),
                      if (state.isGenerating) ...[
                        const SizedBox(width: 10),
                        OutlinedButton.icon(
                          style: OutlinedButton.styleFrom(
                            foregroundColor: theme.colorScheme.error,
                            side: BorderSide(color: theme.colorScheme.error),
                            padding: const EdgeInsets.symmetric(
                              horizontal: 14,
                              vertical: 12,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(8),
                            ),
                          ),
                          icon: const Icon(Icons.stop_rounded, size: 18),
                          label: const Text(
                            _kStopLabel,
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          onPressed: () => widget.controller.dispatch(
                            const StopGenerationIntent(),
                          ),
                        ),
                      ],
                    ],
                  ),
                  TextButton.icon(
                    icon: const Icon(Icons.clear_all_rounded, size: 16),
                    label: const Text(
                      _kClearLabel,
                      style: TextStyle(fontSize: 12),
                    ),
                    onPressed: () {
                      setState(() {
                        _textController.clear();
                      });
                    },
                  ),
                ],
              ),
              const SizedBox(height: 24),

              // Generated Audio History
              if (audioTurns.isNotEmpty) ...[
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Flexible(
                      child: Text(
                        _kOutputsHeader,
                        overflow: TextOverflow.ellipsis,
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0.8,
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ),
                    TextButton.icon(
                      icon: const Icon(Icons.delete_sweep_rounded, size: 14),
                      onPressed: () => widget.controller.dispatch(
                        const ClearTranscriptIntent(),
                      ),
                      label: const Text(
                        _kClearOutputsLabel,
                        style: TextStyle(fontSize: 11),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Column(
                  children: [
                    for (int i = 0; i < audioTurns.length; i++) ...[
                      if (i > 0) const SizedBox(height: 12),
                      Builder(
                        builder: (context) {
                          final turn = audioTurns[audioTurns.length - 1 - i];
                          final stats = turn.stats;
                          return Container(
                            padding: const EdgeInsets.all(14),
                            decoration: BoxDecoration(
                              color: theme.colorScheme.surface,
                              borderRadius: BorderRadius.circular(10),
                              border: Border.all(
                                color: theme.colorScheme.outlineVariant,
                              ),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                if (turn.audioDurationSeconds != null ||
                                    turn.audioSamples != null)
                                  AudioWaveformBubble(
                                    durationSeconds:
                                        turn.audioDurationSeconds ?? 0,
                                    samples: turn.audioSamples,
                                    audioPlayer: widget.controller.audioPlayer,
                                  )
                                else if (turn.isGenerating)
                                  Row(
                                    children: [
                                      const SizedBox(
                                        width: 14,
                                        height: 14,
                                        child: CircularProgressIndicator(
                                          strokeWidth: 2,
                                        ),
                                      ),
                                      const SizedBox(width: 8),
                                      Expanded(
                                        child: Text(
                                          turn.statusText ??
                                              _kDefaultSynthesizingStatus,
                                          maxLines: 1,
                                          overflow: TextOverflow.ellipsis,
                                          style: TextStyle(
                                            fontSize: 12,
                                            color: theme
                                                .colorScheme
                                                .onSurfaceVariant,
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                if (turn.text.isNotEmpty) ...[
                                  const SizedBox(height: 8),
                                  Text(
                                    turn.text,
                                    style: TextStyle(
                                      fontSize: 13,
                                      color: theme.colorScheme.onSurface,
                                      height: 1.4,
                                    ),
                                  ),
                                ],
                                if (stats != null) ...[
                                  const SizedBox(height: 8),
                                  Wrap(
                                    spacing: 12,
                                    runSpacing: 4,
                                    children: [
                                      if (stats.tokens > 0)
                                        Text(
                                          'Tokens: ${stats.tokens}',
                                          style: TextStyle(
                                            fontSize: 10.5,
                                            color: theme
                                                .colorScheme
                                                .onSurfaceVariant,
                                          ),
                                        ),
                                      if (stats.tps > 0)
                                        Text(
                                          'Speed: ${stats.tps.toStringAsFixed(1)} tok/s',
                                          style: TextStyle(
                                            fontSize: 10.5,
                                            color: theme
                                                .colorScheme
                                                .onSurfaceVariant,
                                          ),
                                        ),
                                      if (stats.audioDurationSeconds != null)
                                        Text(
                                          'Audio: ${stats.audioDurationSeconds!.toStringAsFixed(1)}s',
                                          style: TextStyle(
                                            fontSize: 10.5,
                                            fontWeight: FontWeight.w600,
                                            color: theme.colorScheme.primary,
                                          ),
                                        ),
                                      if (stats.audioRtf != null)
                                        Text(
                                          'RTF: ${stats.audioRtf!.toStringAsFixed(2)}x',
                                          style: TextStyle(
                                            fontSize: 10.5,
                                            fontWeight: FontWeight.w600,
                                            color: theme.colorScheme.primary,
                                          ),
                                        ),
                                      Text(
                                        'Latency: ${stats.totalMs}ms',
                                        style: TextStyle(
                                          fontSize: 10.5,
                                          color: theme
                                              .colorScheme
                                              .onSurfaceVariant,
                                        ),
                                      ),
                                      if (turn.text.isNotEmpty)
                                        IconButton(
                                          icon: Icon(
                                            Icons.copy_rounded,
                                            size: 13,
                                            color: theme
                                                .colorScheme
                                                .onSurfaceVariant
                                                .withValues(alpha: 0.7),
                                          ),
                                          padding: EdgeInsets.zero,
                                          constraints: const BoxConstraints(
                                            minWidth: 18,
                                            minHeight: 18,
                                          ),
                                          tooltip: AppStrings.copyTextTooltip,
                                          onPressed: () {
                                            Clipboard.setData(
                                              ClipboardData(text: turn.text),
                                            );
                                            ScaffoldMessenger.of(
                                              context,
                                            ).showSnackBar(
                                              const SnackBar(
                                                content: Text(
                                                  AppStrings.copiedToClipboard,
                                                ),
                                                duration: Duration(seconds: 1),
                                                behavior:
                                                    SnackBarBehavior.floating,
                                              ),
                                            );
                                          },
                                        ),
                                    ],
                                  ),
                                ],
                              ],
                            ),
                          );
                        },
                      ),
                    ],
                  ],
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _VoiceChip extends StatelessWidget {
  const _VoiceChip({
    required this.label,
    required this.selected,
    required this.onSelected,
  });

  final String label;
  final bool selected;
  final VoidCallback onSelected;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return ChoiceChip(
      label: Text(label, style: const TextStyle(fontSize: 11.5)),
      selected: selected,
      onSelected: (_) => onSelected(),
      selectedColor: theme.colorScheme.primary.withValues(alpha: 0.18),
      backgroundColor: theme.colorScheme.surface,
      side: BorderSide(
        color: selected
            ? theme.colorScheme.primary
            : theme.colorScheme.outlineVariant,
      ),
      labelStyle: TextStyle(
        fontSize: 11.5,
        fontWeight: selected ? FontWeight.w600 : FontWeight.normal,
        color: selected
            ? theme.colorScheme.primary
            : theme.colorScheme.onSurface,
      ),
    );
  }
}
