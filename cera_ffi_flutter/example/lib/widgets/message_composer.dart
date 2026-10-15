import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../app_constants.dart';
import '../services/audio_recorder_service.dart';
import 'audio_waveform.dart';

/// Bottom message composer with text input, image attachment preview,
/// vision picker trigger, audio push-to-talk microphone trigger, and send/stop button.
class MessageComposer extends StatefulWidget {
  const MessageComposer({
    super.key,
    required this.controller,
    required this.isBusy,
    required this.isGenerating,
    required this.canAttachImage,
    required this.canAttachAudio,
    required this.pendingImageBytes,
    required this.pendingImageName,
    required this.onSend,
    required this.onStop,
    required this.onPickImage,
    required this.onClearImage,
    required this.onSendAudio,
    this.audioRecorder,
  });

  final TextEditingController controller;
  final bool isBusy;
  final bool isGenerating;
  final bool canAttachImage;
  final bool canAttachAudio;
  final Uint8List? pendingImageBytes;
  final String? pendingImageName;
  final VoidCallback onSend;
  final VoidCallback onStop;
  final VoidCallback onPickImage;
  final VoidCallback onClearImage;
  final void Function(List<double> pcm, int sampleRate) onSendAudio;
  final AudioRecorderService? audioRecorder;

  @override
  State<MessageComposer> createState() => _MessageComposerState();
}

class _MessageComposerState extends State<MessageComposer> {
  static const _kDefaultImageName = 'Attached image';
  static const _kPendingImageSubtitle = 'Will be sent with next prompt';
  static const _kRemoveImageTooltip = 'Remove attached image';
  static const _kAttachImageTooltip = 'Attach image for vision prompt';
  static const _kGeneratingHint = 'Model is generating response...';
  static const _kIdleHint = 'Ask anything or attach media...';
  static const _kStopTooltip = 'Stop generation';
  static const _kSendTooltip = 'Send message';
  static const double _kCancelDragThreshold = -40.0;

  AudioRecorderService? _localRecorder;

  AudioRecorderService get _audioRecorder =>
      widget.audioRecorder ?? (_localRecorder ??= AudioRecorderService());

  bool _isStartingRecording = false;
  bool _isRecordingAudio = false;
  bool _draggedToCancel = false;
  Offset? _pointerOrigin;
  int _recordingSeconds = 0;
  Timer? _recordTimer;

  @override
  void didUpdateWidget(MessageComposer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.audioRecorder != oldWidget.audioRecorder &&
        widget.audioRecorder != null) {
      _localRecorder?.dispose();
      _localRecorder = null;
    }
  }

  @override
  void dispose() {
    _recordTimer?.cancel();
    if (_isRecordingAudio || _isStartingRecording) {
      _audioRecorder.cancelRecording().ignore();
    }
    _localRecorder?.dispose();
    super.dispose();
  }

  Future<void> _startRecording(Offset globalPosition) async {
    if (widget.isBusy || _isRecordingAudio || _isStartingRecording) return;
    _isStartingRecording = true;
    _pointerOrigin = globalPosition;
    try {
      await _audioRecorder.startRecording(
        sampleRate: ChatDefaults.recordingSampleRate,
      );
      if (!mounted) {
        await _audioRecorder.cancelRecording();
        return;
      }
      if (!_isStartingRecording) {
        // User already released finger before recording finished starting
        await _audioRecorder.cancelRecording();
        return;
      }
      _isStartingRecording = false;
      setState(() {
        _isRecordingAudio = true;
        _draggedToCancel = false;
        _recordingSeconds = 0;
      });
      _recordTimer?.cancel();
      _recordTimer = Timer.periodic(const Duration(seconds: 1), (_) {
        if (mounted && _isRecordingAudio) {
          setState(() => _recordingSeconds++);
        }
      });
    } catch (err) {
      _isStartingRecording = false;
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Microphone error: $err')));
      }
    }
  }

  void _onPointerMove(Offset currentPosition) {
    final origin = _pointerOrigin;
    if (!_isRecordingAudio || origin == null) return;
    final diff = currentPosition - origin;
    // Dragging left or upwards by more than 40 pixels triggers cancel
    final isCancel =
        diff.dx < _kCancelDragThreshold || diff.dy < _kCancelDragThreshold;
    if (isCancel != _draggedToCancel) {
      setState(() => _draggedToCancel = isCancel);
    }
  }

  Future<void> _stopRecording() async {
    if (_isStartingRecording) {
      _isStartingRecording = false;
      _pointerOrigin = null;
      try {
        await _audioRecorder.cancelRecording();
      } catch (_) {}
      return;
    }
    if (!_isRecordingAudio) return;
    _recordTimer?.cancel();
    final cancelled = _draggedToCancel;
    _pointerOrigin = null;

    setState(() {
      _isRecordingAudio = false;
      _draggedToCancel = false;
    });

    if (cancelled) {
      try {
        await _audioRecorder.cancelRecording();
      } catch (_) {}
      return;
    }

    try {
      final pcm = await _audioRecorder.stopRecording();
      if (pcm.isNotEmpty && mounted) {
        widget.onSendAudio(pcm, ChatDefaults.recordingSampleRate);
      }
    } catch (err) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Recording error: $err')));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final pendingBytes = widget.pendingImageBytes;
    return Container(
      decoration: BoxDecoration(
        color: theme.scaffoldBackgroundColor,
        border: Border(
          top: BorderSide(color: theme.colorScheme.outlineVariant, width: 1),
        ),
      ),
      padding: const EdgeInsets.fromLTRB(16, 10, 16, 14),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 840),
          child: Container(
            decoration: BoxDecoration(
              color: theme.colorScheme.surface,
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: theme.colorScheme.outline),
            ),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (pendingBytes != null) ...[
                  Padding(
                    padding: const EdgeInsets.fromLTRB(6, 4, 4, 8),
                    child: Row(
                      children: [
                        ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: Image.memory(
                            pendingBytes,
                            width: 44,
                            height: 44,
                            fit: BoxFit.cover,
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                widget.pendingImageName ?? _kDefaultImageName,
                                style: TextStyle(
                                  fontSize: 13,
                                  fontWeight: FontWeight.w600,
                                  color: theme.colorScheme.onSurface,
                                ),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                              ),
                              Text(
                                _kPendingImageSubtitle,
                                style: TextStyle(
                                  fontSize: 11,
                                  color: theme.colorScheme.onSurfaceVariant,
                                ),
                              ),
                            ],
                          ),
                        ),
                        IconButton(
                          icon: Icon(
                            Icons.close_rounded,
                            size: 18,
                            color: theme.colorScheme.onSurfaceVariant,
                          ),
                          tooltip: _kRemoveImageTooltip,
                          onPressed: widget.isBusy ? null : widget.onClearImage,
                        ),
                      ],
                    ),
                  ),
                  Divider(color: theme.colorScheme.outlineVariant, height: 1),
                  const SizedBox(height: 4),
                ],
                Row(
                  children: [
                    if (widget.canAttachImage && !_isRecordingAudio) ...[
                      IconButton(
                        icon: Icon(
                          Icons.add_photo_alternate_outlined,
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                        tooltip: _kAttachImageTooltip,
                        onPressed: widget.isBusy ? null : widget.onPickImage,
                      ),
                    ],
                    if (widget.canAttachAudio) ...[
                      Listener(
                        onPointerDown: (event) =>
                            _startRecording(event.position),
                        onPointerMove: (event) =>
                            _onPointerMove(event.position),
                        onPointerUp: (_) => _stopRecording(),
                        onPointerCancel: (_) => _stopRecording(),
                        child: Container(
                          decoration: BoxDecoration(
                            color: _isRecordingAudio
                                ? (_draggedToCancel
                                      ? theme.colorScheme.error
                                      : theme.colorScheme.primary)
                                : theme.colorScheme.primary.withValues(
                                    alpha: 0.1,
                                  ),
                            shape: BoxShape.circle,
                          ),
                          padding: const EdgeInsets.all(8),
                          child: Icon(
                            _isRecordingAudio
                                ? Icons.mic_rounded
                                : Icons.mic_none_rounded,
                            color: _isRecordingAudio
                                ? theme.colorScheme.onPrimary
                                : theme.colorScheme.primary,
                            size: 20,
                          ),
                        ),
                      ),
                      const SizedBox(width: 6),
                    ],
                    Expanded(
                      child: _isRecordingAudio
                          ? _buildRecordingIndicator(theme)
                          : TextField(
                              controller: widget.controller,
                              style: TextStyle(
                                fontSize: 14,
                                color: theme.colorScheme.onSurface,
                              ),
                              decoration: InputDecoration(
                                hintText: widget.isGenerating
                                    ? _kGeneratingHint
                                    : _kIdleHint,
                                hintStyle: TextStyle(
                                  fontSize: 14,
                                  color: theme.colorScheme.onSurfaceVariant,
                                ),
                                isDense: true,
                                contentPadding: const EdgeInsets.symmetric(
                                  horizontal: 10,
                                  vertical: 10,
                                ),
                                border: InputBorder.none,
                                enabledBorder: InputBorder.none,
                                focusedBorder: InputBorder.none,
                              ),
                              readOnly: widget.isGenerating,
                              onSubmitted: (_) => widget.onSend(),
                            ),
                    ),
                    const SizedBox(width: 6),
                    if (widget.isGenerating)
                      IconButton.filled(
                        style: IconButton.styleFrom(
                          backgroundColor: theme.colorScheme.error,
                          foregroundColor: theme.colorScheme.onError,
                          minimumSize: const Size(36, 36),
                          padding: EdgeInsets.zero,
                        ),
                        icon: const Icon(Icons.stop_rounded, size: 19),
                        tooltip: _kStopTooltip,
                        onPressed: widget.onStop,
                      )
                    else if (!_isRecordingAudio)
                      IconButton.filled(
                        style: IconButton.styleFrom(
                          backgroundColor: theme.colorScheme.primary,
                          foregroundColor: theme.colorScheme.onPrimary,
                          minimumSize: const Size(36, 36),
                          padding: EdgeInsets.zero,
                        ),
                        icon: const Icon(Icons.arrow_upward_rounded, size: 19),
                        tooltip: _kSendTooltip,
                        onPressed: widget.isBusy ? null : widget.onSend,
                      ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildRecordingIndicator(ThemeData theme) {
    final color = _draggedToCancel
        ? theme.colorScheme.error
        : theme.colorScheme.primary;

    return RecordingWaveformIndicator(
      recordingSeconds: _recordingSeconds,
      isCancelled: _draggedToCancel,
      color: color,
    );
  }
}
