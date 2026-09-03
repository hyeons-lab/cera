import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
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
  late final TextEditingController _textController;

  static String _getModelDisplayName(ChatState state) {
    final model = state.loadedModel;
    if (model is BundleModelSource &&
        model.displayName != null &&
        model.displayName!.isNotEmpty) {
      return model.displayName!;
    }
    final rawName = model?.name;
    if (rawName == null || rawName.isEmpty) {
      return 'LFM2';
    }
    var name = rawName;
    if (name.toLowerCase().endsWith('.gguf')) {
      name = name.substring(0, name.length - 5);
    }
    if (name.contains(' · ')) {
      name = name.split(' · ').first;
    }
    return name;
  }

  static String _defaultSampleText(String modelName) =>
      'Hello, this voice was synthesized entirely on-device with the $modelName model powered by Cera.';

  static List<String> _getSamplePrompts(String modelName) => [
    _defaultSampleText(modelName),
    'Cera runs high-performance multimodal AI on Apple Silicon, WebGPU, and CPU.',
    'On-device intelligence guarantees complete privacy with zero cloud latency.',
    'Streaming ISTFT produces 24 kHz neural audio in real-time without external TTS libraries.',
  ];

  @override
  void initState() {
    super.initState();
    final modelName = _getModelDisplayName(widget.state);
    _textController = TextEditingController(
      text: _defaultSampleText(modelName),
    );
  }

  @override
  void didUpdateWidget(TtsStudioView oldWidget) {
    super.didUpdateWidget(oldWidget);
    final oldModel = _getModelDisplayName(oldWidget.state);
    final newModel = _getModelDisplayName(widget.state);
    if (oldModel != newModel) {
      final oldDefault = _defaultSampleText(oldModel);
      if (_textController.text.isEmpty || _textController.text == oldDefault) {
        _textController.text = _defaultSampleText(newModel);
      }
    }
  }

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

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

    // Find turns that contain assistant audio outputs
    final audioTurns = state.turns
        .where(
          (t) =>
              t.role == 'assistant' &&
              (t.audioSamples != null ||
                  t.audioDurationSeconds != null ||
                  t.isGenerating),
        )
        .toList();

    return SingleChildScrollView(
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
                            'Neural Text-to-Speech Studio',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                              color: theme.colorScheme.onSurface,
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            'Direct on-device speech synthesis via neural vocoder (no external TTS library).',
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
                      color: theme.colorScheme.primary.withValues(alpha: 0.08),
                      borderRadius: BorderRadius.circular(6),
                      border: Border.all(
                        color: theme.colorScheme.primary.withValues(alpha: 0.2),
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
                            'Vocoder Active: 24,000 Hz Float32 · ${state.backend ?? "GPU"}',
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
                      color: theme.colorScheme.error.withValues(alpha: 0.08),
                      borderRadius: BorderRadius.circular(6),
                      border: Border.all(
                        color: theme.colorScheme.error.withValues(alpha: 0.25),
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
                            'An audio model (like LFM2.5-Audio) is required for neural TTS.',
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
                            'Load Audio Model',
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
            'SAMPLE SCRIPTS',
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
              children: _getSamplePrompts(_getModelDisplayName(state)).map((
                prompt,
              ) {
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
                    side: BorderSide(color: theme.colorScheme.outlineVariant),
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
            'INPUT TEXT TO SPEAK',
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
            style: TextStyle(fontSize: 14, color: theme.colorScheme.onSurface),
            decoration: InputDecoration(
              hintText: 'Enter text to synthesize into speech...',
              hintStyle: TextStyle(color: theme.colorScheme.onSurfaceVariant),
              filled: true,
              fillColor: theme.colorScheme.surface,
              contentPadding: const EdgeInsets.all(14),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: BorderSide(color: theme.colorScheme.outlineVariant),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(10),
                borderSide: BorderSide(color: theme.colorScheme.outlineVariant),
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
          // Voice Persona Selector
          Text(
            'VOICE PERSONA',
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
              _VoiceChip(
                label: '👩 US Female (Default)',
                selected:
                    state.settings.ttsStudioVoice == 'Use the US female voice.',
                onSelected: () => widget.controller.dispatch(
                  const UpdateSettingsIntent(
                    ttsStudioVoice: 'Use the US female voice.',
                  ),
                ),
              ),
              _VoiceChip(
                label: '👨 US Male',
                selected:
                    state.settings.ttsStudioVoice == 'Use the US male voice.',
                onSelected: () => widget.controller.dispatch(
                  const UpdateSettingsIntent(
                    ttsStudioVoice: 'Use the US male voice.',
                  ),
                ),
              ),
              _VoiceChip(
                label: '👩 UK Female',
                selected:
                    state.settings.ttsStudioVoice == 'Use the UK female voice.',
                onSelected: () => widget.controller.dispatch(
                  const UpdateSettingsIntent(
                    ttsStudioVoice: 'Use the UK female voice.',
                  ),
                ),
              ),
              _VoiceChip(
                label: '👨 UK Male',
                selected:
                    state.settings.ttsStudioVoice == 'Use the UK male voice.',
                onSelected: () => widget.controller.dispatch(
                  const UpdateSettingsIntent(
                    ttsStudioVoice: 'Use the UK male voice.',
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),

          // Action Buttons
          Row(
            children: [
              FilledButton.icon(
                style: FilledButton.styleFrom(
                  backgroundColor: theme.colorScheme.primary,
                  foregroundColor: theme.colorScheme.onPrimary,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 20,
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
                  state.isGenerating ? 'Synthesizing...' : 'Synthesize Speech',
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                onPressed: (state.isBusy || !hasAudioOut) ? null : _synthesize,
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
                    'Stop',
                    style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
                  ),
                  onPressed: () =>
                      widget.controller.dispatch(const StopGenerationIntent()),
                ),
              ],
              const Spacer(),
              TextButton.icon(
                icon: const Icon(Icons.clear_all_rounded, size: 16),
                label: const Text('Clear', style: TextStyle(fontSize: 12)),
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
                Text(
                  'SYNTHESIZED SPEECH OUTPUTS',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.8,
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
                TextButton.icon(
                  icon: const Icon(Icons.delete_sweep_rounded, size: 14),
                  onPressed: () =>
                      widget.controller.dispatch(const ClearTranscriptIntent()),
                  label: const Text(
                    'Clear Outputs',
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
                                durationSeconds: turn.audioDurationSeconds ?? 0,
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
                                  Text(
                                    turn.statusText ?? 'Synthesizing audio...',
                                    style: TextStyle(
                                      fontSize: 12,
                                      color: theme.colorScheme.onSurfaceVariant,
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
                            if (turn.stats != null) ...[
                              const SizedBox(height: 8),
                              Wrap(
                                spacing: 12,
                                runSpacing: 4,
                                children: [
                                  if (turn.stats!.tokens > 0)
                                    Text(
                                      'Tokens: ${turn.stats!.tokens}',
                                      style: TextStyle(
                                        fontSize: 10.5,
                                        color:
                                            theme.colorScheme.onSurfaceVariant,
                                      ),
                                    ),
                                  if (turn.stats!.tps > 0)
                                    Text(
                                      'Speed: ${turn.stats!.tps.toStringAsFixed(1)} tok/s',
                                      style: TextStyle(
                                        fontSize: 10.5,
                                        color:
                                            theme.colorScheme.onSurfaceVariant,
                                      ),
                                    ),
                                  if (turn.stats!.audioDurationSeconds != null)
                                    Text(
                                      'Audio: ${turn.stats!.audioDurationSeconds!.toStringAsFixed(1)}s',
                                      style: TextStyle(
                                        fontSize: 10.5,
                                        fontWeight: FontWeight.w600,
                                        color: theme.colorScheme.primary,
                                      ),
                                    ),
                                  if (turn.stats!.audioRtf != null)
                                    Text(
                                      'RTF: ${turn.stats!.audioRtf!.toStringAsFixed(2)}x',
                                      style: TextStyle(
                                        fontSize: 10.5,
                                        fontWeight: FontWeight.w600,
                                        color: theme.colorScheme.primary,
                                      ),
                                    ),
                                  Text(
                                    'Latency: ${turn.stats!.totalMs}ms',
                                    style: TextStyle(
                                      fontSize: 10.5,
                                      color: theme.colorScheme.onSurfaceVariant,
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
                                      tooltip: 'Copy text',
                                      onPressed: () {
                                        Clipboard.setData(
                                          ClipboardData(text: turn.text),
                                        );
                                        ScaffoldMessenger.of(
                                          context,
                                        ).showSnackBar(
                                          const SnackBar(
                                            content: Text(
                                              'Copied to clipboard',
                                            ),
                                            duration: Duration(seconds: 1),
                                            behavior: SnackBarBehavior.floating,
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
