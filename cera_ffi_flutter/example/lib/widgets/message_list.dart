import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../app_constants.dart';
import '../chat_state.dart';
import '../services/audio_player_service.dart';
import 'audio_waveform.dart';

/// Conversation transcript list with message bubbles, stats, and empty state.
class MessageList extends StatelessWidget {
  const MessageList({
    super.key,
    required this.turns,
    required this.scrollController,
    this.audioPlayer,
    this.hasModel = false,
    this.isLoading = false,
    this.onOpenCatalog,
    this.onOpenLocalModel,
    this.onSelectPrompt,
  });

  final List<Turn> turns;
  final ScrollController scrollController;
  final AudioPlayerService? audioPlayer;
  final bool hasModel;
  final bool isLoading;
  final VoidCallback? onOpenCatalog;
  final VoidCallback? onOpenLocalModel;
  final ValueChanged<String>? onSelectPrompt;

  static const _kHeroTitle = 'Cera On-Device AI';
  static const _kHeroSubtitle =
      'High-performance local LLM, Vision, and Neural Speech inference across CPU, Metal, WebGPU, and Hexagon NPU.';
  static const _kPillAccelerated = 'GPU & NPU Accelerated';
  static const _kPillMultimodal = 'Multimodal Vision & Voice';
  static const _kPillPrivate = '100% Local & Private';
  static const _kBrowseCatalogButton = 'Browse Model Catalog';
  static const _kOpenLocalButton = 'Open Local .gguf';
  static const _kTryPromptHeader = 'TRY A PROMPT';
  static const _starterPrompts = [
    'Explain how speculative decoding speeds up local LLMs.',
    'Write a clean Rust function to compute cosine similarity.',
    'Compare CPU, WebGPU, and NPU execution for transformer inference.',
  ];

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    if (turns.isEmpty) {
      return Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 28),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 580),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 68,
                  height: 68,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [
                        theme.colorScheme.primary.withValues(alpha: 0.22),
                        theme.colorScheme.primary.withValues(alpha: 0.06),
                      ],
                    ),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: theme.colorScheme.primary.withValues(alpha: 0.35),
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: theme.colorScheme.primary.withValues(
                          alpha: 0.14,
                        ),
                        blurRadius: 24,
                        spreadRadius: 2,
                      ),
                    ],
                  ),
                  child: Icon(
                    Icons.auto_awesome_rounded,
                    size: 30,
                    color: theme.colorScheme.primary,
                  ),
                ),
                const SizedBox(height: 18),
                Text(
                  _kHeroTitle,
                  style: TextStyle(
                    fontSize: 21,
                    fontWeight: FontWeight.w700,
                    letterSpacing: -0.3,
                    color: theme.colorScheme.onSurface,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  _kHeroSubtitle,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 13.5,
                    color: theme.colorScheme.onSurfaceVariant,
                    height: 1.45,
                  ),
                ),
                const SizedBox(height: 16),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  alignment: WrapAlignment.center,
                  children: const [
                    _FeaturePill(
                      icon: Icons.bolt_rounded,
                      label: _kPillAccelerated,
                    ),
                    _FeaturePill(
                      icon: Icons.visibility_outlined,
                      label: _kPillMultimodal,
                    ),
                    _FeaturePill(
                      icon: Icons.shield_outlined,
                      label: _kPillPrivate,
                    ),
                  ],
                ),
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
                              horizontal: 18,
                              vertical: 12,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10),
                            ),
                          ),
                          onPressed: onOpenCatalog,
                          icon: const Icon(
                            Icons.cloud_download_outlined,
                            size: 17,
                          ),
                          label: const Text(
                            _kBrowseCatalogButton,
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      if (onOpenLocalModel != null)
                        OutlinedButton.icon(
                          style: OutlinedButton.styleFrom(
                            foregroundColor: theme.colorScheme.onSurface,
                            side: BorderSide(color: theme.colorScheme.outline),
                            padding: const EdgeInsets.symmetric(
                              horizontal: 16,
                              vertical: 12,
                            ),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10),
                            ),
                          ),
                          onPressed: onOpenLocalModel,
                          icon: const Icon(
                            Icons.folder_open_outlined,
                            size: 17,
                          ),
                          label: const Text(
                            _kOpenLocalButton,
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ),
                    ],
                  ),
                ],
                if (hasModel && !isLoading && onSelectPrompt != null) ...[
                  const SizedBox(height: 24),
                  Text(
                    _kTryPromptHeader,
                    style: TextStyle(
                      fontSize: 10.5,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 0.9,
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    alignment: WrapAlignment.center,
                    children: [
                      for (final prompt in _starterPrompts)
                        ActionChip(
                          avatar: Icon(
                            Icons.north_east_rounded,
                            size: 13,
                            color: theme.colorScheme.primary,
                          ),
                          label: Text(
                            prompt,
                            style: TextStyle(
                              fontSize: 12,
                              color: theme.colorScheme.onSurface,
                            ),
                          ),
                          backgroundColor: theme.colorScheme.surface,
                          side: BorderSide(
                            color: theme.colorScheme.outlineVariant,
                          ),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(10),
                          ),
                          onPressed: () => onSelectPrompt?.call(prompt),
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
            constraints: const BoxConstraints(maxWidth: 840),
            child: _TurnBubble(turn: turn, audioPlayer: audioPlayer),
          ),
        );
      },
    );
  }
}

class _FeaturePill extends StatelessWidget {
  const _FeaturePill({required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: theme.colorScheme.outlineVariant),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 13, color: theme.colorScheme.primary),
          const SizedBox(width: 5),
          Flexible(
            child: Text(
              label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                fontSize: 11.5,
                fontWeight: FontWeight.w500,
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _TurnBubble extends StatelessWidget {
  const _TurnBubble({required this.turn, this.audioPlayer});

  final Turn turn;
  final AudioPlayerService? audioPlayer;

  static const _kUserGradientEnd = Color(0xFF2563EB);
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
    final imageBytes = turn.imageBytes;
    final audioDuration = turn.audioDurationSeconds;
    final modelName = turn.modelName;
    final stats = turn.stats;

    return Padding(
      padding: const EdgeInsets.only(bottom: 18),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: isUser
            ? MainAxisAlignment.end
            : MainAxisAlignment.start,
        children: [
          if (!isUser) ...[
            Container(
              width: 32,
              height: 32,
              margin: const EdgeInsets.only(top: 2),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    theme.colorScheme.primary.withValues(alpha: 0.22),
                    theme.colorScheme.primary.withValues(alpha: 0.08),
                  ],
                ),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(
                  color: theme.colorScheme.primary.withValues(alpha: 0.3),
                ),
              ),
              child: Center(
                child: Icon(
                  Icons.smart_toy_rounded,
                  size: 16,
                  color: theme.colorScheme.primary,
                ),
              ),
            ),
            const SizedBox(width: 10),
          ],
          Flexible(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 680),
              child: Column(
                crossAxisAlignment: isUser
                    ? CrossAxisAlignment.end
                    : CrossAxisAlignment.start,
                children: [
                  if (imageBytes != null) ...[
                    Container(
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: theme.colorScheme.outline),
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(11),
                        child: Image.memory(
                          imageBytes,
                          width: 230,
                          fit: BoxFit.cover,
                        ),
                      ),
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
                  if (turn.text.isNotEmpty || turn.isGenerating)
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 12,
                      ),
                      decoration: BoxDecoration(
                        gradient: isUser
                            ? LinearGradient(
                                begin: Alignment.topLeft,
                                end: Alignment.bottomRight,
                                colors: [
                                  theme.colorScheme.primary,
                                  _kUserGradientEnd,
                                ],
                              )
                            : null,
                        color: isUser ? null : theme.colorScheme.surface,
                        borderRadius: BorderRadius.circular(16).copyWith(
                          bottomRight: isUser
                              ? const Radius.circular(4)
                              : const Radius.circular(16),
                          bottomLeft: !isUser
                              ? const Radius.circular(4)
                              : const Radius.circular(16),
                        ),
                        border: isUser
                            ? null
                            : Border.all(color: theme.colorScheme.outline),
                      ),
                      child: turn.isGenerating && turn.text.isEmpty
                          ? _TypingIndicator(label: turn.statusText)
                          : SelectableText(
                              turn.text,
                              style: TextStyle(
                                fontSize: 14.5,
                                height: 1.52,
                                color: isUser
                                    ? theme.colorScheme.onPrimary
                                    : theme.colorScheme.onSurface,
                              ),
                            ),
                    ),
                  if (!isUser &&
                      !turn.isGenerating &&
                      (modelName != null ||
                          stats != null ||
                          turn.text.isNotEmpty)) ...[
                    const SizedBox(height: 6),
                    Wrap(
                      crossAxisAlignment: WrapCrossAlignment.center,
                      spacing: 8,
                      runSpacing: 4,
                      children: [
                        if (modelName != null)
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 7,
                              vertical: 2.5,
                            ),
                            decoration: BoxDecoration(
                              color: theme.colorScheme.primary.withValues(
                                alpha: 0.12,
                              ),
                              borderRadius: BorderRadius.circular(6),
                              border: Border.all(
                                color: theme.colorScheme.primary.withValues(
                                  alpha: 0.24,
                                ),
                              ),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(
                                  Icons.memory_rounded,
                                  size: 11,
                                  color: theme.colorScheme.primary,
                                ),
                                const SizedBox(width: 4),
                                Flexible(
                                  child: Text(
                                    modelName,
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                    style: TextStyle(
                                      fontSize: 10.5,
                                      fontWeight: FontWeight.w600,
                                      color: theme.colorScheme.primary,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        if (stats != null)
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 7,
                              vertical: 2.5,
                            ),
                            decoration: BoxDecoration(
                              color: theme.colorScheme.surface,
                              borderRadius: BorderRadius.circular(6),
                              border: Border.all(
                                color: theme.colorScheme.outlineVariant,
                              ),
                            ),
                            child: Wrap(
                              crossAxisAlignment: WrapCrossAlignment.center,
                              spacing: 4,
                              children: [
                                if (stats.tokens > 0)
                                  Text(
                                    '${stats.tokens} tokens${AppStrings.metaSeparator}${stats.tps.toStringAsFixed(1)} tok/s',
                                    style: TextStyle(
                                      fontSize: 11,
                                      fontWeight: FontWeight.w600,
                                      color: theme.colorScheme.onSurfaceVariant,
                                    ),
                                  ),
                                if (stats.audioRtf != null &&
                                    stats.audioDurationSeconds != null) ...[
                                  if (stats.tokens > 0)
                                    Text(
                                      _kBulletSeparator,
                                      style: TextStyle(
                                        fontSize: 11,
                                        color: theme
                                            .colorScheme
                                            .onSurfaceVariant
                                            .withValues(alpha: 0.5),
                                      ),
                                    ),
                                  Text(
                                    '${stats.audioDurationSeconds!.toStringAsFixed(1)}s audio${AppStrings.metaSeparator}${stats.audioRtf!.toStringAsFixed(2)}x RTF',
                                    style: TextStyle(
                                      fontSize: 11,
                                      fontWeight: FontWeight.w600,
                                      color: theme.colorScheme.primary,
                                    ),
                                  ),
                                ],
                                if (stats.ttftMs != null) ...[
                                  Text(
                                    '$_kBulletSeparator ${stats.tokens > 0 ? "TTFT" : "TTFA"} ${stats.ttftMs}ms',
                                    style: TextStyle(
                                      fontSize: 11,
                                      color: theme.colorScheme.onSurfaceVariant
                                          .withValues(alpha: 0.8),
                                    ),
                                  ),
                                ],
                                if (stats.tokens == 0 &&
                                    stats.audioDurationSeconds == null)
                                  Text(
                                    '${stats.totalMs}ms',
                                    style: TextStyle(
                                      fontSize: 11,
                                      fontWeight: FontWeight.w600,
                                      color: theme.colorScheme.onSurfaceVariant,
                                    ),
                                  ),
                              ],
                            ),
                          ),
                        if (turn.text.isNotEmpty)
                          IconButton(
                            icon: Icon(
                              Icons.copy_rounded,
                              size: 13,
                              color: theme.colorScheme.onSurfaceVariant
                                  .withValues(alpha: 0.7),
                            ),
                            padding: EdgeInsets.zero,
                            constraints: const BoxConstraints(
                              minWidth: 22,
                              minHeight: 22,
                            ),
                            tooltip: AppStrings.copyTextTooltip,
                            onPressed: () => _copyToClipboard(context),
                          ),
                      ],
                    ),
                  ],
                  if (isUser && turn.text.isNotEmpty) ...[
                    const SizedBox(height: 4),
                    Row(
                      mainAxisSize: MainAxisSize.min,
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        IconButton(
                          icon: Icon(
                            Icons.copy_rounded,
                            size: 13,
                            color: theme.colorScheme.onSurfaceVariant
                                .withValues(alpha: 0.6),
                          ),
                          padding: EdgeInsets.zero,
                          constraints: const BoxConstraints(
                            minWidth: 20,
                            minHeight: 20,
                          ),
                          tooltip: AppStrings.copyTextTooltip,
                          onPressed: () => _copyToClipboard(context),
                        ),
                      ],
                    ),
                  ],
                ],
              ),
            ),
          ),
          if (isUser) const SizedBox(width: 4),
        ],
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
                fontSize: 13,
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
                  width: 6,
                  height: 6,
                  decoration: BoxDecoration(
                    color: theme.colorScheme.primary.withValues(alpha: opacity),
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
