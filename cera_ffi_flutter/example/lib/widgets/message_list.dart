import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
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
  });

  final List<Turn> turns;
  final ScrollController scrollController;
  final AudioPlayerService? audioPlayer;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    if (turns.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 64,
                height: 64,
                decoration: BoxDecoration(
                  color: theme.colorScheme.surface,
                  shape: BoxShape.circle,
                  border: Border.all(color: theme.colorScheme.outline),
                ),
                child: Icon(
                  Icons.auto_awesome_rounded,
                  size: 28,
                  color: theme.colorScheme.primary,
                ),
              ),
              const SizedBox(height: 16),
              Text(
                'Cera On-Device AI',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  color: theme.colorScheme.onSurface,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'High-performance, local LLM & Vision inference on CPU and GPU.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 13,
                  color: theme.colorScheme.onSurfaceVariant,
                  height: 1.4,
                ),
              ),
            ],
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
        return _TurnBubble(turn: turn, audioPlayer: audioPlayer);
      },
    );
  }
}

class _TurnBubble extends StatelessWidget {
  const _TurnBubble({required this.turn, this.audioPlayer});

  final Turn turn;
  final AudioPlayerService? audioPlayer;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isUser = turn.role == 'user';
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
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
              decoration: BoxDecoration(
                color: theme.colorScheme.primary.withValues(alpha: 0.16),
                shape: BoxShape.circle,
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
            child: Column(
              crossAxisAlignment: isUser
                  ? CrossAxisAlignment.end
                  : CrossAxisAlignment.start,
              children: [
                if (turn.imageBytes != null) ...[
                  ClipRRect(
                    borderRadius: BorderRadius.circular(10),
                    child: Image.memory(
                      turn.imageBytes!,
                      width: 220,
                      fit: BoxFit.cover,
                    ),
                  ),
                  const SizedBox(height: 8),
                ],
                if (turn.audioDurationSeconds != null) ...[
                  Padding(
                    padding: const EdgeInsets.only(bottom: 6),
                    child: AudioWaveformBubble(
                      durationSeconds: turn.audioDurationSeconds!,
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
                      color: isUser
                          ? theme.colorScheme.primary
                          : theme.colorScheme.surface,
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
                              fontSize: 14,
                              height: 1.5,
                              color: isUser
                                  ? theme.colorScheme.onPrimary
                                  : theme.colorScheme.onSurface,
                            ),
                          ),
                  ),
                if (!isUser &&
                    !turn.isGenerating &&
                    (turn.modelName != null ||
                        turn.stats != null ||
                        turn.text.isNotEmpty)) ...[
                  const SizedBox(height: 6),
                  Wrap(
                    crossAxisAlignment: WrapCrossAlignment.center,
                    spacing: 8,
                    runSpacing: 4,
                    children: [
                      if (turn.modelName != null)
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 6,
                            vertical: 2,
                          ),
                          decoration: BoxDecoration(
                            color: theme.colorScheme.primary.withValues(
                              alpha: 0.12,
                            ),
                            borderRadius: BorderRadius.circular(4),
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
                              Text(
                                turn.modelName!,
                                style: TextStyle(
                                  fontSize: 10.5,
                                  fontWeight: FontWeight.w600,
                                  color: theme.colorScheme.primary,
                                ),
                              ),
                            ],
                          ),
                        ),
                      if (turn.stats != null)
                        Wrap(
                          crossAxisAlignment: WrapCrossAlignment.center,
                          spacing: 4,
                          children: [
                            if (turn.stats!.tokens > 0)
                              Text(
                                '${turn.stats!.tokens} tokens · ${turn.stats!.tps.toStringAsFixed(1)} tok/s',
                                style: TextStyle(
                                  fontSize: 11,
                                  fontWeight: FontWeight.w600,
                                  color: theme.colorScheme.onSurfaceVariant,
                                ),
                              ),
                            if (turn.stats!.audioRtf != null &&
                                turn.stats!.audioDurationSeconds != null) ...[
                              if (turn.stats!.tokens > 0)
                                Text(
                                  '·',
                                  style: TextStyle(
                                    fontSize: 11,
                                    color: theme.colorScheme.onSurfaceVariant
                                        .withValues(alpha: 0.5),
                                  ),
                                ),
                              Text(
                                '${turn.stats!.audioDurationSeconds!.toStringAsFixed(1)}s audio · ${turn.stats!.audioRtf!.toStringAsFixed(2)}x RTF',
                                style: TextStyle(
                                  fontSize: 11,
                                  fontWeight: FontWeight.w600,
                                  color: theme.colorScheme.primary,
                                ),
                              ),
                            ],
                            if (turn.stats!.ttftMs != null) ...[
                              Text(
                                '· ${turn.stats!.tokens > 0 ? "TTFT" : "TTFA"} ${turn.stats!.ttftMs}ms',
                                style: TextStyle(
                                  fontSize: 11,
                                  color: theme.colorScheme.onSurfaceVariant
                                      .withValues(alpha: 0.8),
                                ),
                              ),
                            ],
                            if (turn.stats!.tokens == 0 &&
                                turn.stats!.audioDurationSeconds == null)
                              Text(
                                '${turn.stats!.totalMs}ms',
                                style: TextStyle(
                                  fontSize: 11,
                                  fontWeight: FontWeight.w600,
                                  color: theme.colorScheme.onSurfaceVariant,
                                ),
                              ),
                          ],
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
                            minWidth: 20,
                            minHeight: 20,
                          ),
                          tooltip: 'Copy text',
                          onPressed: () {
                            Clipboard.setData(ClipboardData(text: turn.text));
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                content: Text('Copied to clipboard'),
                                duration: Duration(seconds: 1),
                                behavior: SnackBarBehavior.floating,
                              ),
                            );
                          },
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
                          color: theme.colorScheme.onSurfaceVariant.withValues(
                            alpha: 0.6,
                          ),
                        ),
                        padding: EdgeInsets.zero,
                        constraints: const BoxConstraints(
                          minWidth: 20,
                          minHeight: 20,
                        ),
                        tooltip: 'Copy text',
                        onPressed: () {
                          Clipboard.setData(ClipboardData(text: turn.text));
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text('Copied to clipboard'),
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
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (widget.label != null) ...[
          Text(
            widget.label!,
            style: TextStyle(
              fontSize: 13,
              color: theme.colorScheme.onSurfaceVariant,
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
