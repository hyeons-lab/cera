import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../services/audio_player_service.dart';

/// Animated live waveform indicator displayed during audio recording.
class RecordingWaveformIndicator extends StatefulWidget {
  const RecordingWaveformIndicator({
    super.key,
    required this.recordingSeconds,
    required this.isCancelled,
    required this.color,
  });

  final int recordingSeconds;
  final bool isCancelled;
  final Color color;

  @override
  State<RecordingWaveformIndicator> createState() =>
      _RecordingWaveformIndicatorState();
}

class _RecordingWaveformIndicatorState
    extends State<RecordingWaveformIndicator>
    with SingleTickerProviderStateMixin {
  late final AnimationController _animController;

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat();
  }

  @override
  void dispose() {
    _animController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final minutes = (widget.recordingSeconds ~/ 60).toString().padLeft(2, '0');
    final seconds = (widget.recordingSeconds % 60).toString().padLeft(2, '0');

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: widget.color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: widget.color, width: 1),
      ),
      child: Row(
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: widget.color,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 8),
          Text(
            '$minutes:$seconds',
            style: TextStyle(
              fontWeight: FontWeight.w700,
              fontSize: 12,
              color: theme.colorScheme.onSurface,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: AnimatedBuilder(
              animation: _animController,
              builder: (context, _) {
                return CustomPaint(
                  size: const Size(double.infinity, 22),
                  painter: _LiveWaveformPainter(
                    progress: _animController.value,
                    color: widget.color,
                    isCancelled: widget.isCancelled,
                  ),
                );
              },
            ),
          ),
          const SizedBox(width: 8),
          Text(
            widget.isCancelled ? 'Release to cancel' : 'Slide to cancel',
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w500,
              color: widget.color,
            ),
          ),
        ],
      ),
    );
  }
}

class _LiveWaveformPainter extends CustomPainter {
  _LiveWaveformPainter({
    required this.progress,
    required this.color,
    required this.isCancelled,
  });

  final double progress;
  final Color color;
  final bool isCancelled;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 2.5;

    const numBars = 20;
    final spacing = size.width / numBars;
    final midY = size.height / 2;

    for (var i = 0; i < numBars; i++) {
      final x = (i + 0.5) * spacing;
      final phase = (progress * 2 * math.pi) + (i * 0.4);
      final heightRatio = isCancelled
          ? 0.2
          : 0.25 + 0.65 * ((math.sin(phase) + 1) / 2);
      final barHeight = math.max(3.0, size.height * heightRatio);

      canvas.drawLine(
        Offset(x, midY - barHeight / 2),
        Offset(x, midY + barHeight / 2),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _LiveWaveformPainter oldDelegate) {
    return oldDelegate.progress != progress ||
        oldDelegate.color != color ||
        oldDelegate.isCancelled != isCancelled;
  }
}

/// Static / playable waveform bar visualizer for recorded voice turns in chat history.
class AudioWaveformBubble extends StatefulWidget {
  const AudioWaveformBubble({
    super.key,
    required this.durationSeconds,
    this.samples,
    this.audioPlayer,
  });

  final double durationSeconds;
  final List<double>? samples;
  final AudioPlayerService? audioPlayer;

  @override
  State<AudioWaveformBubble> createState() => _AudioWaveformBubbleState();
}

class _AudioWaveformBubbleState extends State<AudioWaveformBubble> {
  bool _isPlaying = false;

  List<double> _computeNormalizedBars(List<double>? pcm, int numBars) {
    if (pcm == null || pcm.isEmpty) {
      // Default placeholder bar heights
      return List.generate(
        numBars,
        (i) => 0.3 + 0.6 * math.sin((i / numBars) * math.pi),
      );
    }

    final bars = List<double>.filled(numBars, 0.0);
    final chunkSize = pcm.length ~/ numBars;
    if (chunkSize == 0) return bars;

    double maxVal = 0.0;
    for (var i = 0; i < numBars; i++) {
      double sum = 0.0;
      final start = i * chunkSize;
      final end = math.min(pcm.length, start + chunkSize);
      for (var j = start; j < end; j++) {
        sum += pcm[j].abs();
      }
      final avg = sum / (end - start);
      bars[i] = avg;
      if (avg > maxVal) maxVal = avg;
    }

    if (maxVal > 1e-4) {
      for (var i = 0; i < numBars; i++) {
        bars[i] = (bars[i] / maxVal).clamp(0.15, 1.0);
      }
    }
    return bars;
  }

  Future<void> _togglePlayback() async {
    final player = widget.audioPlayer;
    final samples = widget.samples;
    if (player == null || samples == null || samples.isEmpty) return;

    if (_isPlaying) {
      player.stop();
      setState(() => _isPlaying = false);
    } else {
      setState(() => _isPlaying = true);
      await player.playPcm(samples, sampleRate: 16000);
      if (mounted) {
        setState(() => _isPlaying = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    const numBars = 24;
    final bars = _computeNormalizedBars(widget.samples, numBars);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: theme.colorScheme.primaryContainer.withValues(alpha: 0.35),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: theme.colorScheme.primary.withValues(alpha: 0.3),
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (widget.samples != null && widget.audioPlayer != null) ...[
            InkWell(
              onTap: _togglePlayback,
              borderRadius: BorderRadius.circular(20),
              child: Container(
                width: 28,
                height: 28,
                decoration: BoxDecoration(
                  color: theme.colorScheme.primary,
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  _isPlaying ? Icons.stop_rounded : Icons.play_arrow_rounded,
                  size: 18,
                  color: theme.colorScheme.onPrimary,
                ),
              ),
            ),
            const SizedBox(width: 8),
          ],
          SizedBox(
            width: 110,
            height: 24,
            child: CustomPaint(
              painter: _StaticWaveformPainter(
                bars: bars,
                color: theme.colorScheme.primary,
              ),
            ),
          ),
          const SizedBox(width: 8),
          Text(
            '${widget.durationSeconds.toStringAsFixed(1)}s',
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w600,
              color: theme.colorScheme.primary,
            ),
          ),
        ],
      ),
    );
  }
}

class _StaticWaveformPainter extends CustomPainter {
  _StaticWaveformPainter({required this.bars, required this.color});

  final List<double> bars;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 2.0;

    final spacing = size.width / bars.length;
    final midY = size.height / 2;

    for (var i = 0; i < bars.length; i++) {
      final x = (i + 0.5) * spacing;
      final barHeight = math.max(3.0, size.height * bars[i]);
      canvas.drawLine(
        Offset(x, midY - barHeight / 2),
        Offset(x, midY + barHeight / 2),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _StaticWaveformPainter oldDelegate) {
    return oldDelegate.bars != bars || oldDelegate.color != color;
  }
}
