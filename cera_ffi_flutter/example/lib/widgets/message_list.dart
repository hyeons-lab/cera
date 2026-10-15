import 'package:cera_ffi_flutter/cera_ffi_flutter.dart'
    show CeraAudioProfile, CeraTtsVoice;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../app_constants.dart';
import '../chat_state.dart';
import '../services/audio_player_service.dart';
import 'audio_waveform.dart';

/// Conversation transcript list with borderless editorial assistant responses,
/// soft tonal user pills, and a clean Material 3 empty state.
class MessageList extends StatelessWidget {
  const MessageList({
    super.key,
    required this.turns,
    required this.scrollController,
    this.audioPlayer,
    this.hasModel = false,
    this.isLoading = false,
    this.statusText,
    this.activeBackend,
    this.isTtsMode = false,
    this.ttsSampleTexts = const <String>[],
    this.audioProfile,
    this.selectedVoicePrompt = '',
    this.onVoiceChanged,
    this.onOpenCatalog,
    this.onOpenLocalModel,
    this.onSelectPrompt,
  });

  final List<Turn> turns;
  final ScrollController scrollController;
  final AudioPlayerService? audioPlayer;
  final bool hasModel;
  final bool isLoading;
  final String? statusText;
  final String? activeBackend;
  final bool isTtsMode;
  final List<String> ttsSampleTexts;
  final CeraAudioProfile? audioProfile;
  final String selectedVoicePrompt;
  final ValueChanged<String?>? onVoiceChanged;
  final VoidCallback? onOpenCatalog;
  final VoidCallback? onOpenLocalModel;
  final ValueChanged<String>? onSelectPrompt;

  static const _kReadyGreeting = 'What can I help you with?';
  static const _kTtsReadyGreeting =
      'Enter text below or pick a sample script to synthesize speech.';
  static const _kBrowseCatalogButton = 'Browse Models';
  static const _kOpenLocalButton = 'Open .gguf File';
  static const _kVoicePersonaLabel = 'VOICE PERSONA';
  static const _kDefaultStarterPrompts = [
    'Explain how speculative decoding speeds up local LLMs.',
    'Write a clean Rust function to compute cosine similarity.',
    'Compare CPU, WebGPU, and NPU execution for transformer inference.',
  ];
  static const _kDefaultTtsSamples = [
    'Hello, this voice was synthesized entirely on-device with Cera.',
    'The quick brown fox jumps over the lazy dog near the riverbank.',
    'Local neural speech synthesis runs privately with zero cloud latency.',
  ];

  List<String> get _activePrompts {
    if (isTtsMode) {
      return ttsSampleTexts.isNotEmpty ? ttsSampleTexts : _kDefaultTtsSamples;
    }
    return _kDefaultStarterPrompts;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    if (turns.isEmpty) {
      final subtitle = !hasModel
          ? (statusText ?? AppStrings.initialStatus)
          : (isTtsMode ? _kTtsReadyGreeting : _kReadyGreeting);
      final voices = audioProfile?.voices ?? const <CeraTtsVoice>[];
      final activeVoicePrompt =
          audioProfile?.voiceFor(selectedVoicePrompt)?.prompt ?? '';

      return Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 620),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 52,
                  height: 52,
                  decoration: BoxDecoration(
                    color: theme.colorScheme.surfaceContainerHigh,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Icon(
                    isTtsMode
                        ? Icons.graphic_eq_rounded
                        : Icons.auto_awesome_rounded,
                    size: 24,
                    color: theme.colorScheme.onSurface,
                  ),
                ),
                const SizedBox(height: 16),
                Text(
                  AppStrings.appTitle,
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w700,
                    letterSpacing: -0.4,
                    color: theme.colorScheme.onSurface,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  subtitle,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 14.5,
                    color: theme.colorScheme.onSurfaceVariant,
                    height: 1.45,
                  ),
                ),
                if (hasModel &&
                    activeBackend != null &&
                    activeBackend!.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 5,
                    ),
                    decoration: BoxDecoration(
                      color: theme.colorScheme.surfaceContainerLow,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          Icons.memory_rounded,
                          size: 13,
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                        const SizedBox(width: 5),
                        Flexible(
                          child: Text(
                            activeBackend!,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: TextStyle(
                              fontSize: 11.5,
                              fontWeight: FontWeight.w600,
                              color: theme.colorScheme.onSurfaceVariant,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
                if (hasModel &&
                    !isLoading &&
                    voices.isNotEmpty &&
                    onVoiceChanged != null) ...[
                  const SizedBox(height: 20),
                  Text(
                    _kVoicePersonaLabel,
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
                    alignment: WrapAlignment.center,
                    children: [
                      for (final voice in voices)
                        ChoiceChip(
                          label: Text(
                            voice.label,
                            style: const TextStyle(fontSize: 12.5),
                          ),
                          selected: activeVoicePrompt == voice.prompt,
                          onSelected: (_) => onVoiceChanged?.call(voice.prompt),
                        ),
                    ],
                  ),
                ],
                if (!hasModel &&
                    !isLoading &&
                    (onOpenCatalog != null || onOpenLocalModel != null)) ...[
                  const SizedBox(height: 24),
                  Wrap(
                    spacing: 10,
                    runSpacing: 10,
                    alignment: WrapAlignment.center,
                    children: [
                      if (onOpenCatalog != null)
                        FilledButton.icon(
                          style: FilledButton.styleFrom(
                            backgroundColor: theme.colorScheme.primary,
                            foregroundColor: theme.colorScheme.onPrimary,
                            padding: const EdgeInsets.symmetric(
                              horizontal: 20,
                              vertical: 13,
                            ),
                            shape: const StadiumBorder(),
                            elevation: 0,
                          ),
                          onPressed: onOpenCatalog,
                          icon: const Icon(
                            Icons.cloud_download_outlined,
                            size: 17,
                          ),
                          label: const Text(
                            _kBrowseCatalogButton,
                            style: TextStyle(
                              fontSize: 13.5,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      if (onOpenLocalModel != null)
                        FilledButton.tonalIcon(
                          style: FilledButton.styleFrom(
                            backgroundColor:
                                theme.colorScheme.surfaceContainerHigh,
                            foregroundColor: theme.colorScheme.onSurface,
                            padding: const EdgeInsets.symmetric(
                              horizontal: 18,
                              vertical: 13,
                            ),
                            shape: const StadiumBorder(),
                            elevation: 0,
                          ),
                          onPressed: onOpenLocalModel,
                          icon: const Icon(
                            Icons.folder_open_outlined,
                            size: 17,
                          ),
                          label: const Text(
                            _kOpenLocalButton,
                            style: TextStyle(
                              fontSize: 13.5,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ),
                    ],
                  ),
                ],
                if (hasModel && !isLoading && onSelectPrompt != null) ...[
                  const SizedBox(height: 24),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    alignment: WrapAlignment.center,
                    children: [
                      for (final prompt in _activePrompts)
                        _StarterPromptPill(
                          prompt: prompt,
                          icon: isTtsMode
                              ? Icons.volume_up_rounded
                              : Icons.north_east_rounded,
                          onTap: () => onSelectPrompt?.call(prompt),
                        ),
                    ],
                  ),
                ],
              ],
            ),
          ),
        ),
      );
    }

    return ListView.builder(
      controller: scrollController,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 20),
      itemCount: turns.length,
      itemBuilder: (context, index) {
        final turn = turns[index];
        return Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 768),
            child: _TurnBubble(
              turn: turn,
              audioPlayer: audioPlayer,
              activeBackend: activeBackend,
            ),
          ),
        );
      },
    );
  }
}

class _StarterPromptPill extends StatelessWidget {
  const _StarterPromptPill({
    required this.prompt,
    required this.icon,
    required this.onTap,
  });

  final String prompt;
  final IconData icon;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Material(
      color: theme.colorScheme.surfaceContainerLow,
      borderRadius: BorderRadius.circular(18),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(18),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 14, color: theme.colorScheme.onSurfaceVariant),
              const SizedBox(width: 8),
              Flexible(
                child: Text(
                  prompt,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontSize: 13,
                    color: theme.colorScheme.onSurface,
                    height: 1.35,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _TurnBubble extends StatelessWidget {
  const _TurnBubble({required this.turn, this.audioPlayer, this.activeBackend});

  final Turn turn;
  final AudioPlayerService? audioPlayer;
  final String? activeBackend;

  static const _kBulletSeparator = '·';

  void _copyToClipboard(BuildContext context) {
    Clipboard.setData(ClipboardData(text: turn.text));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text(AppStrings.copiedToClipboard),
        duration: Duration(seconds: 1),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isUser = turn.role == TurnRole.user;

    if (isUser) {
      return _buildUserTurn(context, theme);
    }
    return _buildAssistantTurn(context, theme);
  }

  Widget _buildUserTurn(BuildContext context, ThemeData theme) {
    final imageBytes = turn.imageBytes;
    final audioDuration = turn.audioDurationSeconds;

    return Padding(
      padding: const EdgeInsets.only(bottom: 20, left: 40),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          if (imageBytes != null) ...[
            ClipRRect(
              borderRadius: BorderRadius.circular(16),
              child: Image.memory(imageBytes, width: 220, fit: BoxFit.cover),
            ),
            const SizedBox(height: 8),
          ],
          if (audioDuration != null) ...[
            Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: AudioWaveformBubble(
                durationSeconds: audioDuration,
                samples: turn.audioSamples,
                audioPlayer: audioPlayer,
              ),
            ),
          ],
          if (turn.text.isNotEmpty)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 11),
              decoration: BoxDecoration(
                color: theme.colorScheme.surfaceContainerHigh,
                borderRadius: BorderRadius.circular(20),
              ),
              child: SelectableText(
                turn.text,
                style: TextStyle(
                  fontSize: 15,
                  height: 1.48,
                  color: theme.colorScheme.onSurface,
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildAssistantTurn(BuildContext context, ThemeData theme) {
    final audioDuration = turn.audioDurationSeconds;
    final modelName = turn.modelName;
    final backend = activeBackend;
    final stats = turn.stats;

    return Padding(
      padding: const EdgeInsets.only(bottom: 24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 22,
                height: 22,
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainerHigh,
                  borderRadius: BorderRadius.circular(7),
                ),
                child: Icon(
                  Icons.auto_awesome_rounded,
                  size: 13,
                  color: theme.colorScheme.onSurface,
                ),
              ),
              const SizedBox(width: 8),
              Flexible(
                child: Text(
                  modelName ?? AppStrings.appTitle,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
              ),
              if (backend != null && backend.isNotEmpty) ...[
                const SizedBox(width: 6),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 7,
                    vertical: 2,
                  ),
                  decoration: BoxDecoration(
                    color: theme.colorScheme.surfaceContainerLow,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    backend,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      fontSize: 10.5,
                      fontWeight: FontWeight.w600,
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ),
              ],
            ],
          ),
          const SizedBox(height: 8),
          if (audioDuration != null) ...[
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: AudioWaveformBubble(
                durationSeconds: audioDuration,
                samples: turn.audioSamples,
                audioPlayer: audioPlayer,
              ),
            ),
          ],
          if (turn.isGenerating && turn.text.isEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: _TypingIndicator(label: turn.statusText),
            )
          else if (turn.text.isNotEmpty)
            SelectableText(
              turn.text,
              style: TextStyle(
                fontSize: 15.5,
                height: 1.6,
                color: theme.colorScheme.onSurface,
              ),
            ),
          if (!turn.isGenerating &&
              (stats != null || turn.text.isNotEmpty)) ...[
            const SizedBox(height: 6),
            Wrap(
              crossAxisAlignment: WrapCrossAlignment.center,
              spacing: 10,
              runSpacing: 4,
              children: [
                if (turn.text.isNotEmpty)
                  InkWell(
                    onTap: () => _copyToClipboard(context),
                    borderRadius: BorderRadius.circular(6),
                    child: Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 4,
                        vertical: 2,
                      ),
                      child: Icon(
                        Icons.content_copy_rounded,
                        size: 14,
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ),
                if (stats != null) _buildStatsLine(theme, stats),
              ],
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildStatsLine(ThemeData theme, TurnStats stats) {
    final parts = <String>[];
    if (stats.tokens > 0) {
      parts.add('${stats.tps.toStringAsFixed(1)} tok/s');
      parts.add('${stats.tokens} tok');
    }
    if (stats.audioRtf != null && stats.audioDurationSeconds != null) {
      parts.add('${stats.audioDurationSeconds!.toStringAsFixed(1)}s audio');
      parts.add('${stats.audioRtf!.toStringAsFixed(2)}x RTF');
    }
    if (stats.ttftMs != null) {
      final label = stats.tokens > 0 ? 'TTFT' : 'TTFA';
      parts.add('${stats.ttftMs}ms $label');
    } else if (stats.tokens == 0 && stats.audioDurationSeconds == null) {
      parts.add('${stats.totalMs}ms');
    }

    return Text(
      parts.join(' $_kBulletSeparator '),
      style: TextStyle(
        fontSize: 11.5,
        color: theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.8),
      ),
    );
  }
}

class _TypingIndicator extends StatefulWidget {
  const _TypingIndicator({this.label});

  final String? label;

  @override
  State<_TypingIndicator> createState() => _TypingIndicatorState();
}

class _TypingIndicatorState extends State<_TypingIndicator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _anim = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 1000),
  )..repeat();

  @override
  void dispose() {
    _anim.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final label = widget.label;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (label != null) ...[
          Flexible(
            child: Text(
              label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                fontSize: 13.5,
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
          const SizedBox(width: 8),
        ],
        AnimatedBuilder(
          animation: _anim,
          builder: (context, _) {
            final t = _anim.value;
            return Row(
              mainAxisSize: MainAxisSize.min,
              children: List.generate(3, (i) {
                final offset = (t - (i * 0.2)) % 1.0;
                final opacity = (1.0 - (offset * 0.8)).clamp(0.2, 1.0);
                return Container(
                  margin: const EdgeInsets.symmetric(horizontal: 2),
                  width: 5.5,
                  height: 5.5,
                  decoration: BoxDecoration(
                    color: theme.colorScheme.onSurfaceVariant.withValues(
                      alpha: opacity,
                    ),
                    shape: BoxShape.circle,
                  ),
                );
              }),
            );
          },
        ),
      ],
    );
  }
}
